import invoke


@invoke.task(optional=["problem"])
def test(_, problem=None):
    """
    Tests/verifies correctness of code
    """
    import unittest
    from pathlib import Path

    if problem is None:
        unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"))
    else:
        unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.discover(Path("tests") / "public" / problem)
        )


@invoke.task(optional=["apply"])
def lint(ctx, apply=False):
    """
    Lints code and provides suggestions.

    Args:
        apply (bool, optional): Apply changes in suggestions of isort and black.
            USE ONLY AT THE END OF WHOLE HOMEWORK ASSIGNMENT, as it might result in deletion of important imports.
            Defaults to False.
    """
    ctx.run("flake8 homeworks", echo=True)

    isort_flags = ["--diff", "--check-only"]
    black_flags = ["--diff"]

    if apply:
        isort_flags = black_flags = []

    isort_flags = " ".join(isort_flags)
    black_flags = " ".join(black_flags)

    ctx.run(
        f"isort {isort_flags} homeworks tests", echo=True
    )
    result = ctx.run(
        f"black {black_flags} homeworks tests", echo=True
    )
    if "reformatted" not in result.stderr:
        ctx.run("mypy --no-incremental --cache-dir /dev/null homeworks", echo=True)


@invoke.task
def submit(_):
    """
    Generates a .zip file to be uploaded to gradescope for code submission.
    """
    import shutil
    from pathlib import Path
    import datetime

    # Using __file__ trick makes skips dependency on where task is called
    homework_path = Path(__file__).parent / "homeworks"
    tmp_dir_path = Path(__file__).parent / "tmp_submission_dir"
    zip_file_path = Path(__file__).parent / f"submission_{datetime.datetime.now().strftime('%y-%m-%d-%H_%M_%S')}"

    # Cleanup previous submissions attempts
    if tmp_dir_path.exists():
        shutil.rmtree(tmp_dir_path)
    if zip_file_path.exists():
        os.remove(zip_file_path)

    # Copy tree over to temporary directory
    shutil.copytree(homework_path, tmp_dir_path, ignore=shutil.ignore_patterns("*.pyc", "tmp", "__pycache__", ".mypy_cache"))

    # Zip
    shutil.make_archive(zip_file_path, "zip", tmp_dir_path)

    # Cleanup
    shutil.rmtree(tmp_dir_path)

