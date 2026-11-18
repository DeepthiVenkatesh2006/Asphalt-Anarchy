"""Manual in-process API verification; importing this file never starts servers.

Use scripts/run_demo.py for live deployment. This check is not cross-device proof.
Default partition is student_test; evaluator_test must be explicitly requested.
"""


def main():
    import sys
    from scripts.verify_demo import main as verify
    if '--partition' not in sys.argv:
        sys.argv.extend(['--partition','student_test'])
    verify()


if __name__ == '__main__':
    main()
