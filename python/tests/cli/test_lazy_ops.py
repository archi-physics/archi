"""`archi ops` is registered without importing its module, so other `archi` commands do not pay for it."""
import subprocess
import sys
import textwrap


def run(code: str) -> str:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=True).stdout.strip()


def test_building_the_parser_without_ops_does_not_import_the_ops_module():
    out = run("""
        import sys
        from archi.bundles.cli import build_parser
        build_parser(load_ops=False)
        print('archi.cli.ops' in sys.modules, 'archi.ops.loop' in sys.modules, 'archi.ops.okg_io' in sys.modules)
    """)
    assert out == "False False False"


def test_the_stub_lists_ops_in_help_but_main_loads_the_real_parser_for_ops():
    out = run("""
        import io, sys, contextlib
        from archi.bundles.cli import build_parser, main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                build_parser(load_ops=False).parse_args(['--help'])
            except SystemExit:
                pass
        print('ops' in buf.getvalue(), 'archi.cli.ops' in sys.modules)
        try:
            main(['ops', '--help'])
        except SystemExit:
            pass
        print('MARK', 'archi.cli.ops' in sys.modules)
    """)
    assert out.splitlines()[0] == "True False" and out.splitlines()[-1] == "MARK True"


def test_a_non_ops_command_does_not_load_the_ops_module():
    out = run("""
        import sys, io, contextlib
        from archi.bundles.cli import main
        try:
            main(['bundles', 'list', '--json'])
        except SystemExit:
            pass
        print('MARK', 'archi.cli.ops' in sys.modules)
    """)
    assert out.splitlines()[-1] == "MARK False"


def test_the_full_parser_still_parses_ops_subcommands():
    from archi.bundles.cli import build_parser

    args = build_parser().parse_args(["ops", "pending"])
    assert args.command == "ops" and args.command_name == "pending"
