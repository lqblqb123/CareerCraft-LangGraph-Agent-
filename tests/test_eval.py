"""评测脚本 —— 用 20 组简历+JD 测 Agent 匹配准确率。

用法：python tests/test_eval.py
需要：测试数据文件 tests/eval_cases.json
"""

import json
import time
from pathlib import Path

from app.agents.architect import ResumeOptimizerAgent
from app.agents.requirement import CareerAdvisorAgent
from app.agents.reviewer import ResumeReviewerAgent
from app.config.llm import create_llm


def load_cases() -> list[dict]:
    """从 JSON 文件加载测试用例。"""
    path = Path(__file__).parent / "eval_cases.json"
    if not path.exists():
        print(f"❌ 找不到 {path}，请先运行生成脚本把测试数据存到这里")
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def classify(score: float) -> tuple[str, bool]:
    """分类 + 标记是否为边界值（距离阈值 < 0.05 算边界）。"""
    if score >= 0.65:
        return "high", False
    elif score >= 0.55:
        return "high", True  # 边界 high，接近 medium
    elif score >= 0.40:
        return "medium", False
    elif score >= 0.30:
        return "medium", True  # 边界 medium，接近 low
    else:
        return "low", False


def count_accuracy(details: list[dict]) -> dict:
    """计算严格正确率和宽松正确率（边界不计错）。"""
    strict = sum(1 for d in details if d["correct"])
    relaxed = sum(
        1 for d in details if d["correct"] or d["borderline"]
    )
    return {
        "strict": strict / len(details),
        "relaxed": relaxed / len(details),
        "total": len(details),
        "strict_count": strict,
        "relaxed_count": relaxed,
    }


def run_eval():
    cases = load_cases()
    if not cases:
        return

    llm = create_llm()
    advisor = CareerAdvisorAgent(llm)

    results = {"details": [], "total_time": 0}

    for case in cases:
        start = time.time()
        advisor_result = advisor.invoke(
            raw_requirement=case["resume"],
            requirement=case["jd"],
        )
        predicted, borderline = classify(advisor_result.match_score)
        correct = predicted == case["expected_match"]
        elapsed = time.time() - start
        results["total_time"] += elapsed

        results["details"].append({
            "id": case["id"],
            "expected": case["expected_match"],
            "predicted": predicted,
            "score": round(advisor_result.match_score, 2),
            "correct": correct,
            "borderline": borderline,
            "questions_count": len(advisor_result.questions),
            "time_s": round(elapsed, 1),
        })

    acc = count_accuracy(results["details"])
    avg_time = results["total_time"] / len(cases)

    # --- 输出报告 ---
    print("=" * 55)
    print("📊 评测结果")
    print("=" * 55)
    print(f"测试用例数：{acc['total']}")
    print(f"严格准确率：{acc['strict']:.0%} ({acc['strict_count']}/{acc['total']})")
    print(f"宽松准确率：{acc['relaxed']:.0%} ({acc['relaxed_count']}/{acc['total']})  "
          f"(边界值不计错)")
    print(f"平均耗时：  {avg_time:.1f}s")
    print()

    # 误判详情（只列非边界的真正错误）
    errors = [d for d in results["details"] if not d["correct"] and not d["borderline"]]
    borderline_cases = [d for d in results["details"] if d["borderline"] and not d["correct"]]

    if errors:
        print("❌ 真正误判（非边界）：")
        for e in errors:
            print(f"  #{e['id']}: 期望={e['expected']}, 预测={e['predicted']}, "
                  f"score={e['score']:.2f}")
    else:
        print("✅ 无真正误判")

    if borderline_cases:
        print()
        print("⚠️  边界模糊（不算错）：")
        for b in borderline_cases:
            print(f"  #{b['id']}: 期望={b['expected']}, 预测={b['predicted']}, "
                  f"score={b['score']:.2f}")

    print()
    print("按匹配等级拆分：")
    for label in ["high", "medium", "low"]:
        subset = [d for d in results["details"] if d["expected"] == label]
        c = sum(1 for d in subset if d["correct"])
        print(f"  {label}: {c}/{len(subset)} 严格正确, "
              f"{sum(1 for d in subset if d['correct'] or d['borderline'])}/{len(subset)} 宽松正确")

    return results


if __name__ == "__main__":
    run_eval()
