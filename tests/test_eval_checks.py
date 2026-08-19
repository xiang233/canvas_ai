"""
验证断言原语的边界行为。

回归两个曾经的指标缺陷：
  1. NoError 把「我查不到」判成失败，与 rag_judge 的 false_refusal
     指标方向相反 —— 同一份材料里一个惩罚、一个奖励同样的行为。
  2. MentionsNone 用朴素子串匹配，"95" 会命中 "1995"、"95.5"，
     抗幻觉断言因此产生假阳性。

纯本地，不打网络，不需要任何 token。

用法：
    python -m tests.test_eval_checks        # 从仓库根目录运行
"""

from evals.checks import MentionsAll, MentionsNone, NoAbstention, NoError

RESULTS = []


def check(name, condition):
    RESULTS.append((name, condition))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")


def main() -> int:
    print("\n断言原语边界验证\n")

    # ---- NoError：只抓系统故障，不抓合法弃权 ----
    ne = NoError()
    ok, _ = ne.run("I could not find any course named Underwater Basket Weaving.", None)
    check("NoError 放行合法弃权（曾经会误判为失败）", ok)

    ok, _ = ne.run("I cannot find that course.", None)
    check("NoError 放行 'i cannot'", ok)

    ok, _ = ne.run("Sorry, that course has no assignments yet.", None)
    check("NoError 放行 'sorry'", ok)

    ok, _ = ne.run("API Request Failed (Status Code 500)", None)
    check("NoError 仍然抓住真实故障", not ok)

    ok, _ = ne.run("Traceback (most recent call last)", None)
    check("NoError 抓住 traceback", not ok)

    # ---- NoAbstention：与 NoError 互补，只在有数据的题上用 ----
    na = NoAbstention()
    ok, _ = na.run("I could not find that information.", None)
    check("NoAbstention 抓住该答没答", not ok)

    ok, _ = na.run("Your grade in Data Mining is 88.5.", None)
    check("NoAbstention 放行正常作答", ok)

    ok, _ = na.run("API Request Failed (Status Code 500)", None)
    check("NoAbstention 不越界管系统故障（那是 NoError 的活）", ok)

    # ---- MentionsNone：词边界 ----
    mn = MentionsNone(["95"])
    ok, _ = mn.run("The course ran in 1995 and had no grade.", None)
    check("MentionsNone: '95' 不再命中 '1995'", ok)

    ok, _ = mn.run("Your score is 95.5 percent.", None)
    check("MentionsNone: '95' 不再命中 '95.5'", ok)

    ok, _ = mn.run("Your score is 95 percent.", None)
    check("MentionsNone: '95' 仍然命中独立的 95", not ok)

    mn2 = MentionsNone(["A+"])
    ok, _ = mn2.run("You got an A+ in that course.", None)
    check("MentionsNone: 'A+' 这类符号结尾的值仍能匹配", not ok)

    ok, _ = mn2.run("You got an A in that course.", None)
    check("MentionsNone: 'A+' 不误命中单独的 A", ok)

    mn3 = MentionsNone(["your grade is"])
    ok, _ = mn3.run("Your grade is 88.", None)
    check("MentionsNone: 多词短语仍能匹配", not ok)

    # ---- MentionsAll：同一套边界，课程名不受影响 ----
    ma = MentionsAll(["Data Mining"])
    ok, _ = ma.run("You are enrolled in Data Mining and Algorithms.", None)
    check("MentionsAll: 正常课程名仍然匹配", ok)

    ok, _ = ma.run("You are enrolled in Algorithms.", None)
    check("MentionsAll: 缺失时仍然失败", not ok)

    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{passed}/{len(RESULTS)} 通过\n")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
