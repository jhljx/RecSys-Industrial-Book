---
name: recsys-learning-coach
description: Build a grounded recommendation-system study plan, explain a topic for a graduate student or new engineer, and track local progress. Use for learning, revision, interview preparation, and deciding what to study next.
---
# Recommendation learning coach
Turn a topic or a complete repository course into an evidence-based learning loop.

1. Run `python3 tools/recsys_agent.py doctor`; build the ignored index if needed.
2. For a full path, use `course list`, `course start <new-engineer|graduate>`, then `course today --minutes N`. Read `docs/learning-agent-guide.md` and the returned unit.
3. For an ad-hoc topic, use `learn "<topic>" --level <level> --minutes <N>`; never invent page references.
4. Ask the learner to answer the unit questions before revealing a correction. Separate book evidence, learner reasoning, and extension material.
5. Run the practice main path and at least one failure path. Preserve commands, exit codes, outputs, cause, and recovery condition in the learner's evidence file.
6. Check the evidence against every unit acceptance criterion. A file's existence is not proof of mastery.
7. Submit through `course submit` or evidence-gated `complete`. Scores below 0.8 or missing practice/failure evidence remain `assessed`, not `completed`.
8. Record concrete learner constraints or misunderstandings with `course feedback`; generate the next bounded task with `course today`.
9. Use `course review` for delayed retrieval and transfer tasks, and `self-eval` for a local machine-readable report.

State belongs only in ignored `.recsys-workspace/`. Never store private employer data or claim `[MOCK]` templates are production integrations.
