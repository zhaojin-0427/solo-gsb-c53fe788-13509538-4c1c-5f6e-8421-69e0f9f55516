"""历史修订与防「串版」回归测试。

场景：同一项目同一 doc_id 导入两版（新版在开头插字），覆盖：
* 文档列表只返回当前项目的全部修订（且不再 500）；
* 无 revision 参数默认最新版；缺失修订/文档明确 404；
* 标注按修订绑定：历史视图只看到旧标注，文本与坐标对齐；
* 历史修订只读：不能在旧版上删标注；
* 旧共识按自己保存的输入快照渲染数据（快照文本是旧文本）；
* 切回最新版后仍可继续编辑、求解，旧报告不变。
"""
import os
import tempfile

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
# 与 tests/test_api.py 的约定一致：仅在外部未指定时才自选临时库，
# 避免同一次 pytest 运行里两个模块在导入期争抢 ADJ_DB_PATH。
os.environ.setdefault("ADJ_DB_PATH", _tmp.name)

from fastapi.testclient import TestClient  # noqa: E402

from app import database  # noqa: E402
from app.main import app  # noqa: E402

database.init_db()
client = TestClient(app)


def _build_project(name):
    pid = client.post("/api/projects", json={"name": name}).json()["id"]
    lab = client.post(f"/api/projects/{pid}/labels",
                      json={"name": "ENT", "color": "#2563eb"}).json()["id"]
    u1 = client.post(f"/api/projects/{pid}/annotators",
                     json={"name": "甲", "weight": 8}).json()["id"]
    u2 = client.post(f"/api/projects/{pid}/annotators",
                     json={"name": "乙", "weight": 5}).json()["id"]
    return pid, lab, u1, u2


def test_document_list_scoped_to_project_and_all_revisions():
    p1, _, _, _ = _build_project("rev-p1")
    p2, _, _, _ = _build_project("rev-p2")

    client.post(f"/api/projects/{p1}/documents", json={
        "doc_id": "d", "content": "abc", "token_bounds": [[0, 1], [1, 2], [2, 3]]})
    # 新版在开头插字
    client.post(f"/api/projects/{p1}/documents", json={
        "doc_id": "d", "content": "Xabc", "token_bounds": [[0, 1], [1, 2], [2, 3], [3, 4]]})
    client.post(f"/api/projects/{p2}/documents", json={
        "doc_id": "d", "content": "zzz", "token_bounds": [[0, 1], [1, 2], [2, 3]]})

    # 列表不再 500，且只含当前项目的两版
    r = client.get(f"/api/projects/{p1}/documents")
    assert r.status_code == 200, r.text
    rows = r.json()
    assert {d["doc_id"] for d in rows} == {"d"}
    assert sorted(d["revision"] for d in rows) == [1, 2]
    assert [d["char_count"] for d in rows] == [3, 4]

    # 另一个项目只看到自己的数据，不串项目
    rows2 = client.get(f"/api/projects/{p2}/documents").json()
    assert len(rows2) == 1 and rows2[0]["char_count"] == 3


def test_get_document_default_latest_explicit_and_missing():
    pid, _, _, _ = _build_project("rev-get")
    client.post(f"/api/projects/{pid}/documents", json={"doc_id": "d", "content": "旧文本"})
    client.post(f"/api/projects/{pid}/documents", json={"doc_id": "d", "content": "开头插入新文本"})

    latest = client.get(f"/api/projects/{pid}/documents/d").json()
    assert latest["revision"] == 2 and latest["content"] == "开头插入新文本"

    old = client.get(f"/api/projects/{pid}/documents/d", params={"revision": 1}).json()
    assert old["revision"] == 1 and old["content"] == "旧文本"

    # 缺失版本 / 缺失文档：明确 404，不静默回退最新版
    r = client.get(f"/api/projects/{pid}/documents/d", params={"revision": 99})
    assert r.status_code == 404 and "revision 99" in r.json()["detail"]
    r = client.get(f"/api/projects/{pid}/documents/missing")
    assert r.status_code == 404


def test_annotations_bound_to_revision_history_readonly_and_snapshot():
    pid, lab, u1, u2 = _build_project("rev-bind")
    # v1：文本 "甲乙丙"，标注 [0,1)="甲"
    client.post(f"/api/projects/{pid}/documents", json={
        "doc_id": "d", "content": "甲乙丙",
        "token_bounds": [[0, 1], [1, 2], [2, 3]]})
    a1 = client.post(f"/api/projects/{pid}/documents/d/annotations", json={
        "annotator_id": u1, "label_id": lab, "start": 0, "end": 1}).json()
    a2 = client.post(f"/api/projects/{pid}/documents/d/annotations", json={
        "annotator_id": u2, "label_id": lab, "start": 1, "end": 2}).json()

    con1 = client.post(f"/api/projects/{pid}/documents/d/consensus", json={}).json()
    assert con1["revision"] == 1
    cid = con1["id"]
    report = client.post(f"/api/projects/{pid}/consensuses/{cid}/report", json={}).json()
    report_version = report["report_version"]

    # v2：开头插字，旧坐标 [0,1) 在新文本里已指向另一个字
    client.post(f"/api/projects/{pid}/documents", json={
        "doc_id": "d", "content": "X甲乙丙",
        "token_bounds": [[0, 1], [1, 2], [2, 3], [3, 4]]})
    # 新标注落在新坐标上（插入字符之后）
    a3 = client.post(f"/api/projects/{pid}/documents/d/annotations", json={
        "annotator_id": u1, "label_id": lab, "start": 1, "end": 2}).json()

    # 不带 revision => 默认最新版（v2），只含新标注
    lst = client.get(f"/api/projects/{pid}/documents/d/annotations").json()
    assert lst["revision"] == 2 and lst["is_latest"] is True
    assert {a["id"] for a in lst["annotations"]} == {a3["id"]}

    # 显式 revision=1 => 旧版文本配旧标注，不串版
    old = client.get(f"/api/projects/{pid}/documents/d/annotations",
                     params={"revision": 1}).json()
    assert old["revision"] == 1 and old["is_latest"] is False
    assert {a["id"] for a in old["annotations"]} == {a1["id"], a2["id"]}

    # 历史修订只读：不能删除旧版标注
    r = client.delete(f"/api/projects/{pid}/annotations/{a1['id']}")
    assert r.status_code == 409 and "只读" in r.json()["detail"]
    # 最新版标注可正常删除（切回最新版可继续编辑）
    assert client.delete(f"/api/projects/{pid}/annotations/{a3['id']}").status_code == 200

    # 旧共识接口自带 v1 快照：内容是旧文本，坐标在该文本上成立
    got = client.get(f"/api/projects/{pid}/consensuses/{cid}").json()
    snap = got["snapshot"]
    assert snap["revision"] == 1
    assert snap["content"] == "甲乙丙"
    assert {a["start"] for a in snap["annotations"]} == {0, 1}

    # 旧报告不变，仍指向 revision 1 的快照
    rep = client.get(f"/api/projects/{pid}/reports/{report_version}").json()
    assert rep["input_snapshot"]["revision"] == 1
    assert rep["input_snapshot"]["content"] == "甲乙丙"

    # 最新版可继续求解，生成的新共识属于 revision 2
    client.post(f"/api/projects/{pid}/documents/d/annotations", json={
        "annotator_id": u1, "label_id": lab, "start": 1, "end": 2})
    client.post(f"/api/projects/{pid}/documents/d/annotations", json={
        "annotator_id": u2, "label_id": lab, "start": 2, "end": 3})
    con2 = client.post(f"/api/projects/{pid}/documents/d/consensus", json={}).json()
    assert con2["revision"] == 2
    assert con2["snapshot"]["content"] == "X甲乙丙"

    # 共识列表里两版按各自修订留存
    listing = client.get(f"/api/projects/{pid}/documents/d/consensuses").json()
    assert [c["revision"] for c in listing] == [1, 2]
