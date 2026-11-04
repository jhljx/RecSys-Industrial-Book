"""Exercise the local coaching loop without grading or executing learner commands."""

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recsys_coach


class CoachTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "learning-paths").mkdir()
        (self.root / "learning-paths/courses.json").write_text(json.dumps({"courses": {
            "new-engineer": {"units": [
                {"id": "ne-01", "title": "召回", "deliverable": "召回方案", "reading": "ch04 PDF 3-12；ch10 PDF 2-23、54-80",
                 "questions": ["为何召回不等于排序？", "如何设计失败路径？"], "acceptance": ["记录延迟与兜底"]},
                {"id": "ne-02", "title": "排序", "deliverable": "排序方案", "reading": "", "questions": []}]}}}), encoding="utf-8")
        (self.root / "ch04 召回").mkdir()
        (self.root / "ch04 召回/book.pdf").touch()

    def test_semantic_feedback_changes_follow_up_without_grading(self):
        expected = {"status": "provisional", "diagnosis": [{"observation": "混淆候选规模与排序目标", "suggestion": "对照召回链路解释候选集", "evidence_ids": [1]}],
                    "question": "召回和排序的输出各是什么？"}
        with patch.object(recsys_coach, "_semantic_feedback", return_value=expected):
            report = recsys_coach.coach(self.root, "new-engineer", feedback="我不理解召回", semantic=True)
        self.assertEqual(expected, report["semantic_feedback"])
        self.assertEqual(expected["question"], report["follow_up"])
        self.assertEqual("needs_evidence", report["status"])
        self.assertFalse((self.root / ".recsys-workspace/progress.jsonl").exists())

    def test_explicit_other_unit_feedback_is_not_misdiagnosed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "learning-paths").mkdir()
            (root / "learning-paths/courses.json").write_text(json.dumps({"courses": {
                "new-engineer": {"units": [{"id": "ne-01", "title": "反馈闭环", "concepts": ["feedback-loop"],
                                             "deliverable": "闭环图", "reading": "ch01 PDF 5-10", "questions": []}]}}}), encoding="utf-8")
            with patch.object(recsys_coach, "_current_unit", return_value={"id": "ne-01", "title": "反馈闭环",
                               "concepts": ["feedback-loop"], "deliverable": "闭环图", "reading": "ch01 PDF 5-10", "questions": []}), patch(
                               "recsys_navigation.resolve", return_value={"matches": [{"id": "negative-sampling"}]}):
                (root / "knowledge").mkdir()
                (root / "knowledge/graph.json").touch()
                (root / "knowledge/paper-map.json").touch()
                result = recsys_coach.coach(root, "new-engineer", feedback="负采样偏差仍不清楚", semantic=True)
            self.assertEqual("topic_mismatch", result["status"])
            self.assertFalse((root / recsys_coach.STATE_FILE).exists())

    def test_feedback_only_and_cross_session_follow_up(self):
        first = recsys_coach.coach(self.root, "new-engineer", feedback="不理解多路召回如何去重")
        self.assertEqual("needs_evidence", first["status"])
        self.assertEqual("为何召回不等于排序？", first["follow_up"])
        second = recsys_coach.coach(self.root, "new-engineer", feedback="已经画出了数据流")
        self.assertEqual(first["follow_up"], second["previous"]["follow_up"])
        state = (self.root / recsys_coach.STATE_FILE).read_text(encoding="utf-8")
        self.assertNotIn("book.pdf", state)
        self.assertNotIn("已经画出了数据流", second["previous"]["next_tasks"])

    def test_structural_diagnosis_and_no_false_grading(self):
        submission = self.root / "evidence.md"
        submission.write_text("## 本书依据\n见教材\n## 我的解释\n学完了\n## 实践\n主路径输出：ok\n失败路径输出：test\n## 问题答案\n知道了\n", encoding="utf-8")
        result = recsys_coach.coach(self.root, "new-engineer", submission)
        self.assertEqual("needs_revision", result["status"])
        self.assertTrue(any("PDF 第 N 页" in finding["action"] for finding in result["diagnosis"]))
        self.assertEqual("ch04 召回/book.pdf", result["book_reading"]["pdf_pointers"][0]["source"])
        self.assertIn("未执行", result["limits"])
        self.assertNotIn("学完了", (self.root / recsys_coach.STATE_FILE).read_text(encoding="utf-8"))

    def test_complete_structure_still_requires_manual_review(self):
        submission = self.root / "evidence.md"
        submission.write_text("## 本书依据\nch04 召回/book.pdf PDF 第 3 页：待人工核对。\n"
                              "## 我的解释\n输入是用户请求，输出是候选集合，需要监控时延与结果规模。\n"
                              "## 实践\n主路径命令：python demo.py\n主路径退出码：0\n主路径输出：returned 3 items\n"
                              "失败路径命令：python demo.py --bad\n失败路径退出码：1\n失败路径输出：invalid input\n"
                              "失败原因：输入格式错误\n恢复条件：改正输入格式\n"
                              "## 问题答案\n1. 召回输出候选集供排序处理，因此两者目标和规模不同。\n"
                              "2. 修改输入观察退出码和输出，确认异常处理路径有效。\n", encoding="utf-8")
        result = recsys_coach.coach(self.root, "new-engineer", submission)
        self.assertEqual("ready_for_manual_review", result["status"])
        self.assertIn("不会修改课程完成状态", result["limits"])
        self.assertFalse((self.root / ".recsys-workspace/progress.jsonl").exists())

    def test_current_unit_ignores_non_course_and_rejects_invalid_inputs(self):
        (self.root / ".recsys-workspace").mkdir()
        (self.root / ".recsys-workspace/progress.jsonl").write_text(
            json.dumps({"action": "submit", "course_id": "new-engineer", "unit_id": "ne-01", "status": "completed"}) + "\n", encoding="utf-8")
        report = recsys_coach.coach(self.root, "new-engineer", feedback="需要继续")
        self.assertEqual("ne-01", report["unit_id"])
        with self.assertRaises(ValueError):
            recsys_coach.coach(self.root, "new-engineer", feedback=" " )
        with self.assertRaises(ValueError):
            recsys_coach.coach(self.root, "not-a-course", feedback="hello")
        with (self.root / ".recsys-workspace/progress.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"record_type": "course", "action": "submit", "course_id": "new-engineer", "unit_id": "ne-01", "status": "completed"}) + "\n")
        self.assertEqual("ne-02", recsys_coach.coach(self.root, "new-engineer", feedback="现在看排序")["unit_id"])


if __name__ == "__main__":
    unittest.main()
