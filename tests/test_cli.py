"""The command line protects the user from the easy mistakes: a wrong path, the same video twice, and reusing an
output folder that holds the results for other videos."""
import json, os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deltanav.__main__ import check_inputs


def _video(tmp_path, name, size):
    p = tmp_path / name
    p.write_bytes(b'\0' * size)
    return str(p)


def test_any_file_names_work_and_the_flags_decide_which_is_which(tmp_path):
    a, b = _video(tmp_path, 'IMG_1234.MOV', 10), _video(tmp_path, 'whatever name.mp4', 20)
    files = check_inputs(str(tmp_path / 'out'), a, b)
    assert files == {'before': ['IMG_1234.MOV', 10], 'after': ['whatever name.mp4', 20]}


def test_wrong_path_and_same_video_twice_stop_with_a_message(tmp_path):
    a = _video(tmp_path, 'a.mov', 10)
    with pytest.raises(SystemExit, match='no such file'):
        check_inputs(str(tmp_path / 'out'), a, str(tmp_path / 'missing.mov'))
    with pytest.raises(SystemExit, match='same file'):
        check_inputs(str(tmp_path / 'out'), a, a)


def test_an_output_folder_is_not_reused_for_other_videos(tmp_path):
    a, b, c = _video(tmp_path, 'a.mov', 10), _video(tmp_path, 'b.mov', 20), _video(tmp_path, 'c.mov', 30)
    out = tmp_path / 'out'; out.mkdir()
    (out / 'run_info.json').write_text(json.dumps({'input_files': check_inputs(str(out), a, b)}))
    assert check_inputs(str(out), a, b)                       # same videos: resume
    with pytest.raises(SystemExit, match='new --out folder'):
        check_inputs(str(out), a, c)                          # other videos: refuse
    assert check_inputs(str(out), a, c, force=True)           # unless asked to replace
