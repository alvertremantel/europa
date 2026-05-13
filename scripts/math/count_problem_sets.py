from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemSet:
    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    @property
    def includes_zero(self) -> bool:
        return self.start <= 0 <= self.end


def count_nonnegative_subtraction_pairs(problem_set: ProblemSet) -> int:
    return problem_set.size * (problem_set.size + 1) // 2


def count_integer_division_pairs(problem_set: ProblemSet) -> int:
    count = 0
    for left in range(problem_set.start, problem_set.end + 1):
        for right in range(problem_set.start, problem_set.end + 1):
            if right != 0 and left % right == 0:
                count += 1
    return count


def count_operations(problem_set: ProblemSet) -> dict[str, int]:
    pair_count = problem_set.size * problem_set.size

    counts = {
        "+": pair_count,
        "-": count_nonnegative_subtraction_pairs(problem_set),
        "*": pair_count,
        "/": count_integer_division_pairs(problem_set),
    }
    counts["total"] = sum(counts.values())
    return counts


def format_range(problem_set: ProblemSet) -> str:
    return f"[{problem_set.start}, {problem_set.end}]"


def main() -> None:
    problem_sets = [
        ProblemSet("small", 0, 20),
        ProblemSet("medium", 21, 100),
        ProblemSet("large", 101, 500),
    ]

    print("Assumptions:")
    print("- Operands are integers.")
    print("- Problems are ordered pairs, so 2 + 3 and 3 + 2 both count.")
    print("- Addition and multiplication allow every ordered pair.")
    print("- Subtraction counts only ordered pairs where left >= right.")
    print(
        "- Division counts only ordered pairs where right != 0 and left % right == 0."
    )
    print()

    grand_total = 0
    for problem_set in problem_sets:
        counts = count_operations(problem_set)
        grand_total += counts["total"]

        print(f"{problem_set.name.upper()} {format_range(problem_set)}")
        print(f"- values in set: {problem_set.size}")
        print(f"- addition: {counts['+']:,}")
        print(f"- subtraction: {counts['-']:,}")
        print(f"- multiplication: {counts['*']:,}")
        print(f"- division: {counts['/']:,}")
        print(f"- total: {counts['total']:,}")
        print()

    print(f"ALL SETS TOTAL: {grand_total:,}")


if __name__ == "__main__":
    main()
