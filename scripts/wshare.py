import gradio as gr
import csv
import logging
import os
import io
import platform
import random
import re
import shutil
import stat
import sys
import tempfile
import time
import modules.extras
import modules.images
import modules.ui
import requests
import json
from itertools import chain
from io import StringIO
from tkinter import Tk, filedialog
from modules import paths, shared, script_callbacks, scripts, images
from modules.shared import opts, cmd_opts
from modules.ui_common import plaintext_to_html
from modules.ui_components import ToolButton, DropdownMulti
from scripts.wshares import wshare_db
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import List, Tuple
from huggingface_hub import whoami, dataset_info, model_info, space_info, upload_file, list_repo_files, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

try:
    from send2trash import send2trash

    send2trash_installed = True
except ImportError:
    print("Share: send2trash is not installed. recycle bin cannot be used.")
    send2trash_installed = False

yappi_do = False

components_list = ["Sort by", "EXIF keyword search", "Generation Info", "File Name", "File Time", "Upload File",
                   "Gallery Controls Bar", "Delete Bar", "Favorite Button"]

num_of_imgs_per_page = 0
loads_files_num = 0
image_ext_list = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]
finfo_aes = {}
exif_cache = {}
finfo_exif = {}
aes_cache = {}
none_select = "Nothing selected"
refresh_symbol = '\U0001f504'  # 🔄
up_symbol = '\U000025b2'  # ▲
down_symbol = '\U000025bc'  # ▼
caution_symbol = '\U000026a0'  # ⚠
folder_symbol = '\U0001f4c2'  # 📂
heart_symbol = '\U00002764'
current_depth = 0
init = True
copy_move = ["Move", "Copy"]
copied_moved = ["Moved", "Copied"]
np = "negative_prompt: "
openoutpaint = False
favorite_tab_name = "Favorites"
path_maps = {
    "txt2img": opts.outdir_samples or opts.outdir_txt2img_samples,
    "img2img": opts.outdir_samples or opts.outdir_img2img_samples,
    "txt2img-grids": opts.outdir_grids or opts.outdir_txt2img_grids,
    "img2img-grids": opts.outdir_grids or opts.outdir_img2img_grids,
    "Extras": opts.outdir_samples or opts.outdir_extras_samples,
    favorite_tab_name: opts.outdir_save

}
default_tab_options = ["txt2img", "img2img", "txt2img-grids", "img2img-grids", "Extras", favorite_tab_name]
tabs_list = [tab.strip() for tab in chain.from_iterable(csv.reader(StringIO(opts.image_browser_active_tabs))) if
             tab] if hasattr(opts, "image_browser_active_tabs") else default_tab_options


class WShareTab():
    seen_base_tags = set()

    def __init__(self, name: str):
        self.name: str = os.path.basename(name) if os.path.isdir(name) else name
        self.path: str = os.path.realpath(path_maps.get(name, name))
        self.base_tag: str = f"wsahre_image_browser_tab_{self.get_unique_base_tag(self.remove_invalid_html_tag_chars(self.name).lower())}"

    def remove_invalid_html_tag_chars(self, tag: str) -> str:
        # Removes any character that is not a letter, a digit, a hyphen, or an underscore
        removed = re.sub(r'[^a-zA-Z0-9\-_]', '', tag)
        return removed

    def get_unique_base_tag(self, base_tag: str) -> str:
        counter = 1
        while base_tag in self.seen_base_tags:
            match = re.search(r'_(\d+)$', base_tag)
            if match:
                counter = int(match.group(1)) + 1
                base_tag = re.sub(r'_(\d+)$', f"_{counter}", base_tag)
            else:
                base_tag = f"{base_tag}_{counter}"
            counter += 1
        self.seen_base_tags.add(base_tag)
        return base_tag

    def __str__(self):
        return f"Name: {self.name} / Path: {self.path} / Base tag: {self.base_tag} / Seen base tags: {self.seen_base_tags}"


tabs_list = [WShareTab(tab) for tab in tabs_list]
# Logging
logger = logging.getLogger(__name__)
logger_mode = logging.ERROR
if hasattr(opts, "image_browser_logger_warning"):
    if opts.image_browser_logger_warning:
        logger_mode = logging.WARNING
if hasattr(opts, "image_browser_logger_debug"):
    if opts.image_browser_logger_debug:
        logger_mode = logging.DEBUG
logger.setLevel(logger_mode)
if (logger.hasHandlers()):
    logger.handlers.clear()
console_handler = logging.StreamHandler()
console_handler.setLevel(logger_mode)
logger.addHandler(console_handler)
# Debug logging
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"{sys.executable} {sys.version}")
    logger.debug(f"{platform.system()} {platform.version()}")
    try:
        git = os.environ.get('GIT', "git")
        commit_hash = os.popen(f"{git} rev-parse HEAD").read()
    except Exception as e:
        commit_hash = e
    logger.debug(f"{commit_hash}")
    logger.debug(f"Gradio {gr.__version__}")
    logger.debug(f"{paths.script_path}")
    with open(cmd_opts.ui_config_file, "r") as f:
        logger.debug(f.read())
    with open(cmd_opts.ui_settings_file, "r") as f:
        logger.debug(f.read())
    logger.debug(os.path.realpath(__file__))
    # logger.debug([str(tab) for tab in tabs_list])


def delete_recycle(filename):
    if opts.image_browser_delete_recycle and send2trash_installed:
        send2trash(filename)
    else:
        file = Path(filename)
        file.unlink()
    return


def img_path_subdirs_get(img_path):
    subdirs = []
    subdirs.append(none_select)
    for item in os.listdir(img_path):
        item_path = os.path.join(img_path, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    return gr.update(choices=subdirs)


def img_path_add_remove(img_dir, path_recorder, add_remove, img_path_depth):
    img_dir = os.path.realpath(img_dir)
    if add_remove == "add" or (add_remove == "remove" and img_dir in path_recorder):
        if add_remove == "add":
            path_recorder[img_dir] = {
                "depth": int(img_path_depth),
                "path_display": f"{img_dir} [{int(img_path_depth)}]"
            }
            wshare_db.update_path_recorder(img_dir, path_recorder[img_dir]["depth"],
                                           path_recorder[img_dir]["path_display"])
        else:
            del path_recorder[img_dir]
            wshare_db.delete_path_recorder(img_dir)
        path_recorder_formatted = [value.get("path_display") for key, value in path_recorder.items()]
        path_recorder_formatted = sorted(path_recorder_formatted, key=lambda x: natural_keys(x.lower()))

    if add_remove == "remove":
        selected = None
    else:
        selected = path_recorder[img_dir]["path_display"]
    return path_recorder, gr.update(choices=path_recorder_formatted, value=selected)


def sort_order_flip(turn_page_switch, sort_order):
    if sort_order == up_symbol:
        sort_order = down_symbol
    else:
        sort_order = up_symbol
    return 1, -turn_page_switch, sort_order


def read_path_recorder():
    path_recorder = wshare_db.load_path_recorder()
    path_recorder_formatted = [value.get("path_display") for key, value in path_recorder.items()]
    path_recorder_formatted = sorted(path_recorder_formatted, key=lambda x: natural_keys(x.lower()))
    path_recorder_unformatted = list(path_recorder.keys())
    path_recorder_unformatted = sorted(path_recorder_unformatted, key=lambda x: natural_keys(x.lower()))

    return path_recorder, path_recorder_formatted, path_recorder_unformatted


def pure_path(path):
    if path == []:
        return path, 0
    match = re.search(r" \[(\d+)\]$", path)
    if match:
        path = path[:match.start()]
        depth = int(match.group(1))
    else:
        depth = 0
    path = os.path.realpath(path)
    return path, depth


def browser2path(img_path_browser):
    img_path, _ = pure_path(img_path_browser)
    return img_path


def totxt(file):
    base, _ = os.path.splitext(file)
    file_txt = base + '.txt'

    return file_txt


def tab_select():
    path_recorder, path_recorder_formatted, path_recorder_unformatted = read_path_recorder()
    return path_recorder, gr.update(choices=path_recorder_unformatted)


def reduplicative_file_move(src, dst):
    def same_name_file(basename, path):
        name, ext = os.path.splitext(basename)
        f_list = os.listdir(path)
        max_num = 0
        for f in f_list:
            if len(f) <= len(basename):
                continue
            f_ext = f[-len(ext):] if len(ext) > 0 else ""
            if f[:len(name)] == name and f_ext == ext:
                if f[len(name)] == "(" and f[-len(ext) - 1] == ")":
                    number = f[len(name) + 1:-len(ext) - 1]
                    if number.isdigit():
                        if int(number) > max_num:
                            max_num = int(number)
        return f"{name}({max_num + 1}){ext}"

    name = os.path.basename(src)
    save_name = os.path.join(dst, name)
    src_txt_exists = False
    if opts.image_browser_txt_files:
        src_txt = totxt(src)
        if os.path.exists(src_txt):
            src_txt_exists = True
    if not os.path.exists(save_name):
        if opts.image_browser_copy_image:
            shutil.copy2(src, dst)
            if opts.image_browser_txt_files and src_txt_exists:
                shutil.copy2(src_txt, dst)
        else:
            shutil.move(src, dst)
            if opts.image_browser_txt_files and src_txt_exists:
                shutil.move(src_txt, dst)
    else:
        name = same_name_file(name, dst)
        if opts.image_browser_copy_image:
            shutil.copy2(src, os.path.join(dst, name))
            if opts.image_browser_txt_files and src_txt_exists:
                shutil.copy2(src_txt, totxt(os.path.join(dst, name)))
        else:
            shutil.move(src, os.path.join(dst, name))
            if opts.image_browser_txt_files and src_txt_exists:
                shutil.move(src_txt, totxt(os.path.join(dst, name)))


def save_image(file_name, filenames, page_index, turn_page_switch, dest_path):
    if file_name is not None and os.path.exists(file_name):
        reduplicative_file_move(file_name, dest_path)
        message = f"<div style='color:#999'>{copied_moved[opts.image_browser_copy_image]} to {dest_path}</div>"
        if not opts.image_browser_copy_image:
            # Force page refresh with checking filenames
            filenames = []
            turn_page_switch = -turn_page_switch
    else:
        message = "<div style='color:#999'>Image not found (may have been already moved)</div>"

    return filenames, page_index, turn_page_switch


def delete_image(name, filenames, image_index, visible_num, turn_page_switch):
    print('python funtion')
    time.sleep(5)
    if name == "":
        return filenames
    else:
        delete_num = 1
        delete_confirm = False
        image_index = int(image_index)
        visible_num = int(visible_num)
        index = list(filenames).index(name)
        new_file_list = []
        if not delete_confirm:
            delete_num = min(visible_num - image_index, delete_num)

        if delete_num > 1:
            # Force refresh page when done, no special js handling necessary
            turn_page_switch = -turn_page_switch
            delete_state = False
        else:
            delete_state = True
        for i, name in enumerate(filenames):
            if i >= index and i < index + delete_num:
                if os.path.exists(name):
                    if opts.image_browser_delete_message:
                        print(f"Deleting file {name}")
                    delete_recycle(name)
                    visible_num -= 1
                    if opts.image_browser_txt_files:
                        txt_file = totxt(name)
                        if os.path.exists(txt_file):
                            delete_recycle(txt_file)
                else:
                    print(f"File does not exist {name}")
            else:
                new_file_list.append(name)
    return new_file_list, delete_state, turn_page_switch, visible_num


def traverse_all_files(curr_path, image_list, tab_base_tag_box, img_path_depth) -> List[
    Tuple[str, os.stat_result, str, int]]:
    global current_depth
    logger.debug(f"curr_path: {curr_path}")
    if curr_path == "":
        return image_list
    f_list = [(os.path.join(curr_path, entry.name), entry.stat()) for entry in os.scandir(curr_path)]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in image_ext_list:
            image_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            if (opts.image_browser_with_subdirs and tab_base_tag_box != "Others") or (
                    tab_base_tag_box == "Others" and img_path_depth != 0 and (
                    current_depth < img_path_depth or img_path_depth < 0)):
                current_depth = current_depth + 1
                image_list = traverse_all_files(fname, image_list, tab_base_tag_box, img_path_depth)
                current_depth = current_depth - 1
    return image_list


def cache_exif(fileinfos):
    global finfo_exif, exif_cache, finfo_aes, aes_cache

    if yappi_do:
        import yappi
        import pandas as pd
        yappi.set_clock_type("wall")
        yappi.start()

    cache_exif_start = time.time()
    new_exif = 0
    new_aes = 0
    conn, cursor = wshare_db.transaction_begin()
    for fi_info in fileinfos:
        if any(fi_info[0].endswith(ext) for ext in image_ext_list):
            found_exif = False
            found_aes = False
            if fi_info[0] in exif_cache:
                finfo_exif[fi_info[0]] = exif_cache[fi_info[0]]
                found_exif = True
            if fi_info[0] in aes_cache:
                finfo_aes[fi_info[0]] = aes_cache[fi_info[0]]
                found_aes = True
            if not found_exif or not found_aes:
                finfo_exif[fi_info[0]] = "0"
                exif_cache[fi_info[0]] = "0"
                finfo_aes[fi_info[0]] = "0"
                aes_cache[fi_info[0]] = "0"
                try:
                    image = Image.open(fi_info[0])
                    (_, allExif, allExif_html) = modules.extras.run_pnginfo(image)
                    image.close()
                except SyntaxError:
                    allExif = False
                    logger.warning(f"Extension and content don't match: {fi_info[0]}")
                except UnidentifiedImageError as e:
                    allExif = False
                    logger.warning(f"UnidentifiedImageError: {e}")
                except PermissionError as e:
                    allExif = False
                    logger.warning(f"PermissionError: {e}: {fi_info[0]}")
                except OSError as e:
                    if e.errno == 22:
                        logger.warning(f"Caught OSError with error code 22: {fi_info[0]}")
                    else:
                        raise
                if allExif:
                    finfo_exif[fi_info[0]] = allExif
                    exif_cache[fi_info[0]] = allExif
                    wshare_db.update_exif_data(conn, fi_info[0], allExif)
                    new_exif = new_exif + 1
                    m = re.search("(?:aesthetic_score:|Score:) (\d+.\d+)", allExif)
                    if m:
                        aes_value = m.group(1)
                    else:
                        aes_value = "0"
                    finfo_aes[fi_info[0]] = aes_value
                    aes_cache[fi_info[0]] = aes_value
                    wshare_db.update_aes_data(conn, fi_info[0], aes_value)
                    new_aes = new_aes + 1
                else:
                    try:
                        filename = os.path.splitext(fi_info[0])[0] + ".txt"
                        geninfo = ""
                        with open(filename) as f:
                            for line in f:
                                geninfo += line
                        finfo_exif[fi_info[0]] = geninfo
                        exif_cache[fi_info[0]] = geninfo
                        wshare_db.update_exif_data(conn, fi_info[0], geninfo)
                        new_exif = new_exif + 1
                        m = re.search("(?:aesthetic_score:|Score:) (\d+.\d+)", geninfo)
                        if m:
                            aes_value = m.group(1)
                        else:
                            aes_value = "0"
                        finfo_aes[fi_info[0]] = aes_value
                        aes_cache[fi_info[0]] = aes_value
                        wshare_db.update_aes_data(conn, fi_info[0], aes_value)
                        new_aes = new_aes + 1
                    except Exception:
                        logger.warning(f"cache_exif: No EXIF in image or txt file for {fi_info[0]}")
                        # Saved with defaults to not scan it again next time
                        finfo_exif[fi_info[0]] = "0"
                        exif_cache[fi_info[0]] = "0"
                        allExif = "0"
                        wshare_db.update_exif_data(conn, fi_info[0], allExif)
                        new_exif = new_exif + 1
                        aes_value = "0"
                        finfo_aes[fi_info[0]] = aes_value
                        aes_cache[fi_info[0]] = aes_value
                        wshare_db.update_aes_data(conn, fi_info[0], aes_value)
                        new_aes = new_aes + 1

    wshare_db.transaction_end(conn, cursor)

    if yappi_do:
        yappi.stop()
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        yappi_stats = yappi.get_func_stats().strip_dirs()
        data = [(s.name, s.ncall, s.tsub, s.ttot, s.ttot / s.ncall) for s in yappi_stats]
        df = pd.DataFrame(data, columns=['name', 'ncall', 'tsub', 'ttot', 'tavg'])
        print(df.to_string(index=False))
        yappi.get_thread_stats().print_all()

    cache_exif_end = time.time()
    logger.debug(
        f"cache_exif: {new_exif}/{len(fileinfos)} cache_aes: {new_aes}/{len(fileinfos)} {round(cache_exif_end - cache_exif_start, 1)} seconds")


def exif_rebuild(maint_wait):
    global finfo_exif, exif_cache, finfo_aes, aes_cache
    if opts.image_browser_scan_exif:
        logger.debug("Rebuild start")
        exif_dirs = wshare_db.get_exif_dirs()
        finfo_aes = {}
        exif_cache = {}
        finfo_exif = {}
        aes_cache = {}
        for key, value in exif_dirs.items():
            if os.path.exists(key):
                print(f"Rebuilding {key}")
                fileinfos = traverse_all_files(key, [], "", 0)
                cache_exif(fileinfos)
        logger.debug("Rebuild end")
        maint_last_msg = "Rebuild finished"
    else:
        maint_last_msg = "Exif cache not enabled in settings"

    return maint_wait, maint_last_msg


def exif_delete_0(maint_wait):
    global finfo_exif, exif_cache, finfo_aes, aes_cache
    if opts.image_browser_scan_exif:
        conn, cursor = wshare_db.transaction_begin()
        wshare_db.delete_exif_0(cursor)
        wshare_db.transaction_end(conn, cursor)
        finfo_aes = {}
        finfo_exif = {}
        exif_cache = wshare_db.load_exif_data(exif_cache)
        aes_cache = wshare_db.load_aes_data(aes_cache)
        maint_last_msg = "Delete finished"
    else:
        maint_last_msg = "Exif cache not enabled in settings"

    return maint_wait, maint_last_msg


def exif_update_dirs(maint_update_dirs_from, maint_update_dirs_to, maint_wait):
    global exif_cache, aes_cache
    if maint_update_dirs_from == "":
        maint_last_msg = "From is empty"
    elif maint_update_dirs_to == "":
        maint_last_msg = "To is empty"
    else:
        maint_update_dirs_from = os.path.realpath(maint_update_dirs_from)
        maint_update_dirs_to = os.path.realpath(maint_update_dirs_to)
        rows = 0
        conn, cursor = wshare_db.transaction_begin()
        wshare_db.update_path_recorder_mult(cursor, maint_update_dirs_from, maint_update_dirs_to)
        rows = rows + cursor.rowcount
        wshare_db.update_exif_data_mult(cursor, maint_update_dirs_from, maint_update_dirs_to)
        rows = rows + cursor.rowcount
        wshare_db.update_ranking_mult(cursor, maint_update_dirs_from, maint_update_dirs_to)
        rows = rows + cursor.rowcount
        wshare_db.transaction_end(conn, cursor)
        if rows == 0:
            maint_last_msg = "No rows updated"
        else:
            maint_last_msg = f"{rows} rows updated. Please reload UI!"

    return maint_wait, maint_last_msg


def reapply_ranking(path_recorder, maint_wait):
    dirs = {}

    for tab in tabs_list:
        if os.path.exists(tab.path):
            dirs[tab.path] = tab.path

    for key in path_recorder:
        if os.path.exists(key):
            dirs[key] = key

    conn, cursor = wshare_db.transaction_begin()

    # Traverse all known dirs, check if missing rankings are due to moved files
    for key in dirs.keys():
        fileinfos = traverse_all_files(key, [], "", 0)
        for (file, _) in fileinfos:
            # Is there a ranking for this full filepath
            ranking_by_file = wshare_db.get_ranking_by_file(cursor, file)
            if ranking_by_file is None:
                name = os.path.basename(file)
                (ranking_by_name, alternate_hash) = wshare_db.get_ranking_by_name(cursor, name)
                # Is there a ranking only for the filename
                if ranking_by_name is not None:
                    hash = wshare_db.get_hash(file)
                    (alternate_file, alternate_ranking) = ranking_by_name
                    if alternate_ranking is not None:
                        (alternate_hash,) = alternate_hash
                    # Does the found filename's file have no hash or the same hash?
                    if alternate_hash is None or hash == alternate_hash:
                        if os.path.exists(alternate_file):
                            # Insert ranking as a copy of the found filename's ranking
                            wshare_db.insert_ranking(cursor, file, alternate_ranking, hash)
                        else:
                            # Replace ranking of the found filename
                            wshare_db.replace_ranking(cursor, file, alternate_file, hash)

    wshare_db.transaction_end(conn, cursor)
    maint_last_msg = "Rankings reapplied"

    return maint_wait, maint_last_msg


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def msg_html(msg, flag):
    if flag:
        return f"<div style='font-size:15px;font-weight:bold;line-height:30px'  align='center'>{msg}</div>"
    else:
        return f"<div style='font-size:15px;font-weight:bold;line-height:30px;color:red'  align='center'>{msg}</div>"


def replicable_push_file(img_path, file_name, token, desc):
    if not (file_name and img_path and token):
        return msg_html("Please enter required information.", False)

    url = "https://replicable.art/api/asset/createV2"
    headers = {'Authorization': 'Bearer {}'.format(token)}
    file_type = file_name.split('.')[-1]
    files = [('file', (file_name, open(img_path, 'rb'), 'image/' + file_type))]
    try:
        res = requests.request("POST", url, headers=headers, files=files, data={'description': desc})
    except Exception as e:
        print(e)
        return msg_html("upload error", False)
    if res.status_code == 201:
        return msg_html("Uploaded successfully!", True)
    elif res.status_code == 401:
        return msg_html("token error", False)
    elif res.status_code == 400:
        return msg_html(json.loads(res.text)['message'], False)
    else:
        return msg_html("upload error", False)


def huggingface_push_file(file_name, file_from, file_to, token):
    if not (file_name and file_to and token):
        return msg_html("Please enter required information.", False)
    repo_type = "dataset"
    branch = "main"
    # check token role
    try:
        try:
            token_role = whoami(token=token)
            if token_role['auth']['accessToken']['role'] == 'read':
                return msg_html("Token type error - make sure the token has WRITE role.", False)
        except RepositoryNotFoundError:
            return msg_html("token error", False)
        # check repo type is model
        try:
            model_info(file_to, token=token)
            return msg_html("Repo type error - make sure the repo is Dataset type.", False)
        except RepositoryNotFoundError:
            pass

        # check repo type is space
        try:
            space_info(file_to, token=token)
            return msg_html("Repo type error - make sure the repo is Dataset type.", False)
        except RepositoryNotFoundError:
            pass

        try:
            dataset_info(file_to, token=token)
        except RepositoryNotFoundError:
            return msg_html(f'"{file_to}" not found', False)
        # check file name
        all_files = list_repo_files(repo_id=file_to, repo_type=repo_type, token=token)
        if file_name in all_files:
            return msg_html("File with the same name exists in the repo.", False)

        # img = Image.open(file_from)
        # img.info['parameters'] = ''
        # img_byte_arr = io.BytesIO()
        # img.save(img_byte_arr, format='PNG')
        # img_byte_arr = img_byte_arr.getvalue()
        try:
            upload_file(path_or_fileobj=file_from, path_in_repo=file_name, revision=branch, repo_id=file_to,
                        commit_message=f"file", token=token, repo_type=repo_type)
        except Exception as e:
            if e.response.status_code == 403:
                return msg_html("Token error - make sure the token has WRITE role.", False)
            elif e.response.status_code == 401:
                return msg_html("Token error - make sure the token has WRITE role.", False)
            else:
                return msg_html("Unknown uploading error - restart and try again", False)

        return msg_html("Uploaded successfully!", True)
    except Exception as e:
        print(e)
        return msg_html("Upload error", False)


def huggingface_push_folder(dir_name, folder_from, folder_to, token):
    print('----')
    print(dir_name + '/' + folder_from)
    if not (folder_from and folder_to and token):
        return msg_html("parameters error or missing", False)
    repo_type = "dataset"
    branch = "main"
    # check token role
    try:
        try:
            token_role = whoami(token=token)
            if token_role['auth']['accessToken']['role'] == 'read':
                return msg_html("token type error,need write role", False)
        except RepositoryNotFoundError:
            return msg_html("token error", False)
        # check repo type is model
        try:
            model_info(folder_to, token=token)
            return msg_html("token type error,need dataset", False)
        except RepositoryNotFoundError:
            pass
        # check repo type is space
        try:
            space_info(folder_to, token=token)
            return msg_html("token type error,need dataset", False)
        except RepositoryNotFoundError:
            pass

        try:
            dataset_info(folder_to, token=token)
        except RepositoryNotFoundError:
            return msg_html(f'"{folder_to}" not found', False)
        # # check file name
        all_files = list_repo_files(repo_id=folder_to, repo_type=repo_type, token=token)
        folder_list = [i.split('/')[0] for i in all_files if '/' in i]

        if folder_from in folder_list:
            return msg_html("folder name already exists", False)
        try:
            upload_folder(folder_path=dir_name + '/' + folder_from, path_in_repo=folder_from, repo_type=repo_type,
                          revision=branch, repo_id=folder_to, token=token)

        except Exception as e:
            if e.response.status_code == 403:
                return msg_html("token type error,need write role", False)
            elif e.response.status_code == 401:
                return msg_html("token error", False)
            else:
                return msg_html("upload error", False)

        return msg_html("uploaded sucessfully!", True)
    except Exception as e:
        print(e)
        return msg_html("upload error", False)


def check_ext(ext):
    found = False
    scripts_list = scripts.list_scripts("scripts", ".py")
    for scriptfile in scripts_list:
        if ext in scriptfile.basedir.lower():
            found = True
            break
    return found


def exif_search(needle, haystack, use_regex, case_sensitive):
    found = False
    if use_regex:
        if case_sensitive:
            pattern = re.compile(needle, re.DOTALL)
        else:
            pattern = re.compile(needle, re.DOTALL | re.IGNORECASE)
        if pattern.search(haystack) is not None:
            found = True
    else:
        if not case_sensitive:
            haystack = haystack.lower()
            needle = needle.lower()
        if needle in haystack:
            found = True
    return found


def get_all_images(dir_name, sort_by, sort_order, keyword, tab_base_tag_box, img_path_depth, ranking_filter,
                   aes_filter_min, aes_filter_max, exif_keyword, negative_prompt_search, use_regex, case_sensitive):
    global current_depth
    current_depth = 0
    fileinfos = traverse_all_files(dir_name, [], tab_base_tag_box, img_path_depth)
    keyword = keyword.strip(" ")

    if opts.image_browser_scan_exif:
        cache_exif(fileinfos)

    if len(keyword) != 0:
        fileinfos = [x for x in fileinfos if keyword.lower() in x[0].lower()]
        filenames = [finfo[0] for finfo in fileinfos]

    if opts.image_browser_scan_exif:
        conn, cursor = wshare_db.transaction_begin()
        if len(exif_keyword) != 0:
            if use_regex:
                regex_error = False
                try:
                    test_re = re.compile(exif_keyword, re.DOTALL)
                except re.error as e:
                    regex_error = True
                    print(f"Regex error: {e}")
            if (use_regex and not regex_error) or not use_regex:
                if negative_prompt_search == "Yes":
                    fileinfos = [x for x in fileinfos if
                                 exif_search(exif_keyword, finfo_exif[x[0]], use_regex, case_sensitive)]
                else:
                    result = []
                    for file_info in fileinfos:
                        file_name = file_info[0]
                        file_exif = finfo_exif[file_name]
                        file_exif_lc = file_exif.lower()
                        start_index = file_exif_lc.find(np)
                        end_index = file_exif.find("\n", start_index)
                        if negative_prompt_search == "Only":
                            start_index = start_index + len(np)
                            sub_string = file_exif[start_index:end_index].strip()
                            if exif_search(exif_keyword, sub_string, use_regex, case_sensitive):
                                result.append(file_info)
                        else:
                            sub_string = file_exif[start_index:end_index].strip()
                            file_exif = file_exif.replace(sub_string, "")

                            if exif_search(exif_keyword, file_exif, use_regex, case_sensitive):
                                result.append(file_info)
                    fileinfos = result
                filenames = [finfo[0] for finfo in fileinfos]
        wshare_db.fill_work_files(cursor, fileinfos)
        if len(aes_filter_min) != 0 or len(aes_filter_max) != 0:
            try:
                aes_filter_min_num = float(aes_filter_min)
            except ValueError:
                aes_filter_min_num = 0
            try:
                aes_filter_max_num = float(aes_filter_max)
            except ValueError:
                aes_filter_max_num = 0
            if aes_filter_min_num < 0:
                aes_filter_min_num = 0
            if aes_filter_max_num <= 0:
                aes_filter_max_num = sys.maxsize

            fileinfos = wshare_db.filter_aes(cursor, fileinfos, aes_filter_min_num, aes_filter_max_num)
            filenames = [finfo[0] for finfo in fileinfos]
        if ranking_filter != "All":
            fileinfos = wshare_db.filter_ranking(cursor, fileinfos, ranking_filter)
            filenames = [finfo[0] for finfo in fileinfos]

        wshare_db.transaction_end(conn, cursor)

    if sort_by == "date":
        if sort_order == up_symbol:
            fileinfos = sorted(fileinfos, key=lambda x: x[1].st_mtime)
        else:
            fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
        filenames = [finfo[0] for finfo in fileinfos]
    elif sort_by == "path name":
        if sort_order == up_symbol:
            fileinfos = sorted(fileinfos)
        else:
            fileinfos = sorted(fileinfos, reverse=True)
        filenames = [finfo[0] for finfo in fileinfos]
    elif sort_by == "random":
        random.shuffle(fileinfos)
        filenames = [finfo[0] for finfo in fileinfos]
    elif sort_by == "ranking":
        finfo_ranked = {}
        for fi_info in fileinfos:
            finfo_ranked[fi_info[0]] = get_ranking(fi_info[0])
        if not down_symbol:
            fileinfos = dict(sorted(finfo_ranked.items(), key=lambda x: (x[1], x[0])))
        else:
            fileinfos = dict(reversed(sorted(finfo_ranked.items(), key=lambda x: (x[1], x[0]))))
        filenames = [finfo for finfo in fileinfos]
    elif sort_by == "aesthetic_score":
        fileinfo_aes = {}
        for finfo in fileinfos:
            fileinfo_aes[finfo[0]] = finfo_aes[finfo[0]]
        if down_symbol:
            fileinfos = dict(reversed(sorted(fileinfo_aes.items(), key=lambda x: (x[1], x[0]))))
        else:
            fileinfos = dict(sorted(fileinfo_aes.items(), key=lambda x: (x[1], x[0])))
        filenames = [finfo for finfo in fileinfos]
    else:
        sort_values = {}
        exif_info = dict(finfo_exif)
        if exif_info:
            for k, v in exif_info.items():
                match = re.search(r'(?<=' + sort_by.lower() + ":" ').*?(?=(,|$))', v.lower())
                if match:
                    sort_values[k] = match.group()
                else:
                    sort_values[k] = "0"
            if down_symbol:
                fileinfos = dict(reversed(sorted(fileinfos, key=lambda x: natural_keys(sort_values[x[0]]))))
            else:
                fileinfos = dict(sorted(fileinfos, key=lambda x: natural_keys(sort_values[x[0]])))
            filenames = [finfo for finfo in fileinfos]
        else:
            filenames = [finfo for finfo in fileinfos]
    return filenames


def get_image_page(img_path, page_index, filenames, sort_by, sort_order, tab_base_tag_box, img_path_depth,
                   exif_keyword, delete_state, hidden):
    ranking_filter, aes_filter_min, aes_filter_max, keyword = "All", "", "", ""
    negative_prompt_search = "No"
    use_regex, case_sensitive = False, False

    msg = "<div style='font-size:15px;font-weight:bold;line-height:30px;color:#FFA500'  align='center'>Select the file first</div>"
    if img_path == "":
        return [], page_index, [], msg, msg, "","", "", "", 0, "", delete_state, None

    # Set temp_dir from webui settings, so gradio uses it
    if shared.opts.temp_dir != "":
        tempfile.tempdir = shared.opts.temp_dir

    img_path, _ = pure_path(img_path)
    if page_index == 1 or page_index == 0 or len(filenames) == 0:
        filenames = get_all_images(img_path, sort_by, sort_order, keyword, tab_base_tag_box, img_path_depth,
                                   ranking_filter, aes_filter_min, aes_filter_max, exif_keyword, negative_prompt_search,
                                   use_regex, case_sensitive)
    page_index = int(page_index)
    length = len(filenames)
    max_page_index = length // num_of_imgs_per_page + 1
    page_index = max_page_index if page_index == -1 else page_index
    page_index = 1 if page_index < 1 else page_index
    page_index = max_page_index if page_index > max_page_index else page_index
    idx_frm = (page_index - 1) * num_of_imgs_per_page
    image_list = filenames[idx_frm:idx_frm + num_of_imgs_per_page]

    visible_num = num_of_imgs_per_page if idx_frm + num_of_imgs_per_page < length else length % num_of_imgs_per_page
    visible_num = num_of_imgs_per_page if visible_num == 0 else visible_num

    load_info = "<div style='color:#999;line-height:64px;' align='center'>"
    load_info += f"{length} images in this directory, divided into {int((length + 1) // num_of_imgs_per_page + 1)} pages"
    load_info += "</div>"

    delete_state = False
    return filenames, gr.update(value=page_index,
                                label=f"Page Index ({page_index}/{max_page_index})"), image_list, msg, msg, "", "", "", "", "", "", visible_num, load_info, delete_state, None


def get_current_file(tab_base_tag_box, num, page_index, filenames):
    file = filenames[int(num) + int((page_index - 1) * num_of_imgs_per_page)]
    return file


def show_image_info(tab_base_tag_box, num, page_index, filenames, turn_page_switch):
    logger.debug(
        f"tab_base_tag_box, num, page_index, len(filenames), num_of_imgs_per_page: {tab_base_tag_box}, {num}, {page_index}, {len(filenames)}, {num_of_imgs_per_page}")
    if len(filenames) == 0:
        # This should only happen if webui was stopped and started again and the user clicks on one of the still displayed images.
        # The state with the filenames will be empty then. In that case we return None to prevent further errors and force a page refresh.
        turn_page_switch = -turn_page_switch
        file = None
        tm = None
        file_name = ""
    else:
        file = filenames[int(num) + int((page_index - 1) * num_of_imgs_per_page)]
        file_name = file.split('/')[-1]
        tm = "<div style='color:#999' align='right'>" + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                      time.localtime(os.path.getmtime(file))) + "</div>"
        with Image.open(file) as image:
            _, geninfo, info = modules.extras.run_pnginfo(image)
    msg = msg_html("Waiting to upload ...", True)
    return msg, msg, file_name, "", file_name,file_name, file, tm, num, file, turn_page_switch, info


def change_dir(img_dir, path_recorder, load_switch, img_path_browser, img_path_depth, img_path):
    warning = None
    img_path, _ = pure_path(img_path)
    img_path_depth_org = img_path_depth
    if img_dir == none_select:
        return warning, gr.update(visible=False), img_path_browser, path_recorder, load_switch, img_path, img_path_depth
    else:
        img_dir, img_path_depth = pure_path(img_dir)
        if warning is None:
            try:
                if os.path.exists(img_dir):
                    try:
                        f = os.listdir(img_dir)
                    except:
                        warning = f"'{img_dir} is not a directory"
                else:
                    warning = "The directory does not exist"
            except:
                warning = "The format of the directory is incorrect"
        if warning is None:
            return "", gr.update(visible=True), img_path_browser, path_recorder, img_dir, img_dir, img_path_depth
        else:
            return warning, gr.update(
                visible=False), img_path_browser, path_recorder, load_switch, img_path, img_path_depth_org


def update_move_text_one(btn):
    btn_text = " ".join(btn.split()[1:])
    return f"{copy_move[opts.image_browser_copy_image]} {btn_text}"


def update_move_text(favorites_btn, to_dir_btn):
    return update_move_text_one(favorites_btn), update_move_text_one(to_dir_btn)


def get_ranking(filename):
    ranking_value = wshare_db.select_ranking(filename)
    return ranking_value


def update_ranking(img_file_name, ranking, img_file_info):
    saved_ranking = get_ranking(img_file_name)
    if saved_ranking != ranking:
        # Update db
        wshare_db.update_ranking(img_file_name, ranking)
        if opts.image_browser_ranking_pnginfo and any(img_file_name.endswith(ext) for ext in image_ext_list[:3]):
            # Update exif
            image = Image.open(img_file_name)
            geninfo, items = images.read_info_from_image(image)
            if geninfo is not None:
                if "Ranking: " in geninfo:
                    if ranking == "None":
                        geninfo = re.sub(r', Ranking: \d+', '', geninfo)
                    else:
                        geninfo = re.sub(r'Ranking: \d+', f'Ranking: {ranking}', geninfo)
                else:
                    geninfo = f'{geninfo}, Ranking: {ranking}'

            original_time = os.path.getmtime(img_file_name)
            images.save_image(image, os.path.dirname(img_file_name), "",
                              extension=os.path.splitext(img_file_name)[1][1:], info=geninfo,
                              forced_filename=os.path.splitext(os.path.basename(img_file_name))[0], save_to_dirs=False)
            os.utime(img_file_name, (original_time, original_time))
            img_file_info = geninfo
    return img_file_info


def create_tab(tab: WShareTab):
    global init, exif_cache, aes_cache, openoutpaint
    dir_name = None
    folder_list = []
    others_dir = False
    standard_ui = True

    if init:
        db_version = wshare_db.check()
        logger.debug(f"db_version: {db_version}")
        exif_cache = wshare_db.load_exif_data(exif_cache)
        aes_cache = wshare_db.load_aes_data(aes_cache)
        init = False

    path_recorder, path_recorder_formatted, path_recorder_unformatted = read_path_recorder()
    openoutpaint = check_ext("openoutpaint")

    if tab.name == "Others":
        others_dir = True
        standard_ui = False
    elif tab.name == "Maintenance":
        maint = True
        standard_ui = False
    else:
        dir_name = tab.path

        folder_list = [i for i in os.listdir(dir_name) if os.path.isdir(dir_name + "/" + i)]

    if standard_ui:
        dir_name = str(Path(dir_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    with gr.Row(visible=others_dir):
        with gr.Column(scale=10):
            img_path = gr.Textbox(dir_name, label="Images directory", placeholder="Input images directory",
                                  interactive=others_dir)
        with gr.Column(scale=1):
            img_path_depth = gr.Number(value="0", label="Sub directory depth")
        with gr.Column(scale=1):
            img_path_save_button = gr.Button(value="Add to / replace in saved directories")

    with gr.Row(visible=others_dir):
        with gr.Column(scale=10):
            img_path_browser = gr.Dropdown(choices=path_recorder_formatted, label="Saved directories")
        with gr.Column(scale=1):
            img_path_remove_button = gr.Button(value="Remove from saved directories")

    with gr.Row(visible=others_dir):
        with gr.Column(scale=10):
            img_path_subdirs = gr.Dropdown(choices=[none_select], value=none_select, label="Sub directories",
                                           interactive=True, elem_id=f"{tab.base_tag}_img_path_subdirs")
        with gr.Column(scale=1):
            img_path_subdirs_button = gr.Button(value="Get sub directories")

    with gr.Row(visible=standard_ui, elem_id=f"{tab.base_tag}_image_browser") as main_panel:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=4):
                        path_recorder = gr.State(path_recorder)
                        warning_box = gr.HTML("<p>&nbsp")
                    with gr.Column(scale=1, visible=False, min_width=150) as favorite_panel:
                        if tab.name != favorite_tab_name:
                            favorites_btn = gr.Button(
                                'Favorites',
                                elem_id=f"{tab.base_tag}_image_browser_favorites_btn")

                    with gr.Column(scale=1, visible=False, min_width=150) as delete_panel:
                        delete = gr.Button('Delete', variant='stop',
                                           elem_id=f"{tab.base_tag}_image_browser_del_img_btn")

                with gr.Row() as gallery_controls_panel:
                    first_page = gr.Button('First Page')
                    prev_page = gr.Button('Prev Page', elem_id=f"{tab.base_tag}_image_browser_prev_page")
                    page_index = gr.Number(value=1, label="Page Index")
                    refresh_index_button = ToolButton(value=refresh_symbol)
                    next_page = gr.Button('Next Page', elem_id=f"{tab.base_tag}_image_browser_next_page")
                    end_page = gr.Button('End Page')

                with gr.Row():
                    image_gallery = gr.Gallery(show_label=False,
                                               elem_id=f"{tab.base_tag}_image_browser_gallery").style(
                        grid=opts.image_browser_page_columns)
                with gr.Row():
                    with gr.Column() as sort_panel:
                        with gr.Row(scale=0.5):
                            sort_by = gr.Dropdown(value="date",
                                                  choices=["date", "cfg scale", "steps", "seed", "sampler", "size",
                                                           "model", "model hash"], label="Sort by")
                            sort_order = ToolButton(value=down_symbol)
                    with gr.Column() as exif_search_panel:
                        exif_keyword_search = gr.Textbox(value="", label="EXIF Keyword Search")
                    with gr.Column() as filename_panel:
                        img_file_path = gr.Textbox(visible=False, interactive=False)
                        img_file_name = gr.Textbox(value="", label="File Name on System", interactive=False)

                with gr.Row() as filetime_panel:
                    img_file_time = gr.HTML()

                with gr.Row() as generation_info_panel:
                    with gr.Accordion("Generation Info", open=False):
                        img_file_info_add = gr.HTML()

            with gr.Column(scale=1) as upload_file_panel:
                with gr.Tab("Replicable"):
                    with gr.Box():
                        with gr.Row().style(equal_height=False):
                            gr.HTML(
                                value="<p>Click <a href=\"https://replicable.art\" style='font-weight:bold;color:#FFA500; display: inline-block'>here</a> to learn how to get an access token.</p></br>")

                        with gr.Row():
                            replicable_token = gr.Textbox(label="Replicable Token", max_lines=1,
                                                          placeholder="Token")
                        gr.HTML("<p>&nbsp")
                        with gr.Row():
                            replicable_desc = gr.Textbox(value="",placeholder='Please provide detailed workflow information if any.', label="Description", max_lines=2,lines=2)
                        gr.HTML("<p>&nbsp")
                        with gr.Row():
                            replicable_file_name = gr.Textbox(value="", label="File Name to Upload", max_lines=1)
                        gr.HTML("<p>&nbsp")
                        with gr.Row():
                            replicable_output = gr.HTML(
                                "<div style='font-size:15px;font-weight:bold;line-height:30px;color:#FFA500'  align='center'>Select the file first</div>")
                        with gr.Row():
                            replicable_put_btn = gr.Button("Upload File to Replicable", variant='primary',
                                                           elem_id="replicable_push_btn")
                            replicable_put_btn.click(fn=lambda: (msg_html("Uploading ...", True)),
                                                     outputs=replicable_output,
                                                     show_progress=False)
                            replicable_put_btn.click(replicable_push_file,
                                                     inputs=[img_file_path, replicable_file_name, replicable_token,replicable_desc],
                                                     outputs=replicable_output, show_progress=False)
                with gr.Tab("HuggingFace"):
                    with gr.Box():
                        with gr.Row().style(equal_height=False):
                            gr.HTML(
                                value="<ul><li>Click <a href=\"https://huggingface.co/docs/hub/security-tokens\" style='font-weight:bold;color:#FFA500; display: inline-block'>here</a> to learn how to get an access token with <div style='font-weight:bold;color:#FFA500; display: inline-block'>'write'</div> role.</li><li>You <div style='font-weight:bold;color:#FFA500; display: inline-block'>MUST include account name</div> in the repo name below, e.g., datamonet/test_repo.</li><li>The repo type must be <a href=\"https://huggingface.co/docs/datasets/upload_dataset#upload-your-filesDataset\" style='font-weight:bold;color:#FFA500; display: inline-block'>Dataset</a>.</li></ul></br>")

                        with gr.Row():
                            text_file_token = gr.Textbox(label="HuggingFace Token", max_lines=1,
                                                         placeholder="Token")
                        gr.HTML("<p>&nbsp")
                        with gr.Row():
                            text_file_to = gr.Textbox(label="Repo Name", max_lines=1, placeholder="Repo Name")
                    with gr.Box():
                        with gr.Row():
                            text_file_name = gr.Textbox(value="", label="File Name to Upload", max_lines=1)

                        with gr.Row():
                            out_file = gr.HTML(
                                "<div style='font-size:15px;font-weight:bold;line-height:30px;color:#FFA500'  align='center'>Select the file first</div>")

                        with gr.Row():
                            btn_push_file = gr.Button("Upload File to HuggingFace", variant='primary',
                                                      elem_id="huggingface_push_btn")
                            btn_push_file.click(fn=lambda: (msg_html("Uploading...", True)),
                                                outputs=out_file, show_progress=False)
                            btn_push_file.click(huggingface_push_file,
                                                inputs=[text_file_name, img_file_path, text_file_to, text_file_token],
                                                outputs=out_file, show_progress=False)
                    with gr.Box():
                        with gr.Row():
                            huggingface_folder_name = gr.Dropdown(value=folder_list[0] if folder_list else "",
                                                                  choices=folder_list, label="Folder Name")

                        with gr.Row():
                            huggingface_folder_out = gr.HTML("<p>&nbsp")
                        with gr.Row():
                            btn_push_folder = gr.Button("Upload Folder to HuggingFace", variant='primary',
                                                        elem_id="huggingface_push_folder_btn")
                            btn_push_folder.click(fn=lambda: (msg_html("Uploading...", True)),
                                                  outputs=huggingface_folder_out, show_progress=False)
                            btn_push_folder.click(huggingface_push_folder,
                                                  inputs=[img_path, huggingface_folder_name, text_file_to,
                                                          text_file_token],
                                                  outputs=huggingface_folder_out, show_progress=False)

                # hidden items
                with gr.Row(visible=False):
                    renew_page = gr.Button("Renew Page", elem_id=f"{tab.base_tag}_image_browser_renew_page")
                    visible_img_num = gr.Number()
                    tab_base_tag_box = gr.Textbox(tab.base_tag)
                    image_index = gr.Textbox(value=-1)
                    set_index = gr.Button('set_index', elem_id=f"{tab.base_tag}_image_browser_set_index")
                    filenames = gr.State([])
                    all_images_list = gr.State()
                    hidden = gr.Image(type="pil")
                    info1 = gr.Textbox()
                    info2 = gr.Textbox()
                    load_switch = gr.Textbox(value="load_switch", label="load_switch")
                    to_dir_load_switch = gr.Textbox(value="to dir load_switch", label="to_dir_load_switch")
                    turn_page_switch = gr.Number(value=1, label="turn_page_switch")
                    img_path_add = gr.Textbox(value="add")
                    img_path_remove = gr.Textbox(value="remove")
                    delete_state = gr.Checkbox(value=False, elem_id=f"{tab.base_tag}_image_browser_delete_state")
                    favorites_path = gr.Textbox(value=opts.outdir_save)
                    mod_keys = ""
                    if opts.image_browser_mod_ctrl_shift:
                        mod_keys = f"{mod_keys}CS"
                    elif opts.image_browser_mod_shift:
                        mod_keys = f"{mod_keys}S"
                    image_browser_mod_keys = gr.Textbox(value=mod_keys,
                                                        elem_id=f"{tab.base_tag}_image_browser_mod_keys")
                    image_browser_prompt = gr.Textbox(elem_id=f"{tab.base_tag}_image_browser_prompt")
                    image_browser_neg_prompt = gr.Textbox(elem_id=f"{tab.base_tag}_image_browser_neg_prompt")

    # Hide components based on opts.image_browser_hidden_components
    hidden_component_map = {
        "Sort by": sort_panel,
        "EXIF keyword search": exif_search_panel,
        "Generation Info": generation_info_panel,
        "File Name": filename_panel,
        "File Time": filetime_panel,
        "Upload File": upload_file_panel,
        "Gallery Controls Bar": gallery_controls_panel,
        "Delete Bar": delete_panel,
        "Favorite Button": favorite_panel,
    }

    if set(hidden_component_map.keys()) != set(components_list):
        logger.warning(
            f"Invalid items present in either hidden_component_map or components_list. Make sure when adding new components they are added to both.")

    override_hidden = set()
    if hasattr(opts, "image_browser_hidden_components"):
        for item in opts.image_browser_hidden_components:
            hidden_component_map[item].visible = False
            override_hidden.add(hidden_component_map[item])

    change_dir_outputs = [warning_box, main_panel, img_path_browser, path_recorder, load_switch, img_path,
                          img_path_depth]
    img_path.submit(change_dir,
                    inputs=[img_path, path_recorder, load_switch, img_path_browser, img_path_depth, img_path],
                    outputs=change_dir_outputs)
    img_path_browser.change(change_dir,
                            inputs=[img_path_browser, path_recorder, load_switch, img_path_browser, img_path_depth,
                                    img_path], outputs=change_dir_outputs)

    # delete
    delete.click(delete_image,
                 inputs=[img_file_path, filenames, image_index, visible_img_num,
                         turn_page_switch],
                 outputs=[filenames, delete_state, turn_page_switch, visible_img_num])
    delete.click(fn=None, _js="wshare_image_browser_delete", inputs=[tab_base_tag_box, image_index], outputs=None)
    if tab.name != favorite_tab_name:
        favorites_btn.click(save_image, inputs=[img_file_path, filenames, page_index, turn_page_switch, favorites_path],
                            outputs=[filenames, page_index, turn_page_switch])

    # turn page
    first_page.click(lambda s: (1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    next_page.click(lambda p, s: (p + 1, -s), inputs=[page_index, turn_page_switch],
                    outputs=[page_index, turn_page_switch])
    prev_page.click(lambda p, s: (p - 1, -s), inputs=[page_index, turn_page_switch],
                    outputs=[page_index, turn_page_switch])
    end_page.click(lambda s: (-1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    load_switch.change(lambda s: (1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    exif_keyword_search.submit(lambda s: (1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    sort_by.change(lambda s: (1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    page_index.submit(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])
    renew_page.click(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])
    refresh_index_button.click(lambda p, s: (p, -s), inputs=[page_index, turn_page_switch],
                               outputs=[page_index, turn_page_switch])
    img_path_depth.change(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])

    turn_page_switch.change(
        fn=get_image_page,
        inputs=[img_path, page_index, filenames, sort_by, sort_order, tab_base_tag_box,
                img_path_depth, exif_keyword_search, delete_state],
        outputs=[filenames, page_index, image_gallery, replicable_output, out_file,
                 replicable_file_name,replicable_desc,
                 text_file_name, img_file_name, img_file_path,
                 img_file_time, visible_img_num, warning_box, delete_state, hidden]

    )
    turn_page_switch.change(fn=None, inputs=[tab_base_tag_box], outputs=None, _js="wshare_image_browser_turnpage")
    turn_page_switch.change(fn=lambda: (gr.update(choices=os.listdir(dir_name))), inputs=[],
                            outputs=[huggingface_folder_name])
    hide_on_thumbnail_view = [delete_panel, favorite_panel, generation_info_panel]
    turn_page_switch.change(
        fn=lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)), inputs=None,
        outputs=hide_on_thumbnail_view)

    sort_order.click(
        fn=sort_order_flip,
        inputs=[turn_page_switch, sort_order],
        outputs=[page_index, turn_page_switch, sort_order]
    )

    # Others
    img_path_subdirs_button.click(
        fn=img_path_subdirs_get,
        inputs=[img_path],
        outputs=[img_path_subdirs]
    )
    img_path_subdirs.change(
        fn=change_dir,
        inputs=[img_path_subdirs, path_recorder, load_switch, img_path_browser, img_path_depth, img_path],
        outputs=change_dir_outputs
    )
    img_path_save_button.click(
        fn=img_path_add_remove,
        inputs=[img_path, path_recorder, img_path_add, img_path_depth],
        outputs=[path_recorder, img_path_browser]
    )
    img_path_remove_button.click(
        fn=img_path_add_remove,
        inputs=[img_path, path_recorder, img_path_remove, img_path_depth],
        outputs=[path_recorder, img_path_browser]
    )

    # other functions
    set_index.click(show_image_info, _js="wshare_image_browser_get_current_img",
                    inputs=[tab_base_tag_box, image_index, page_index, filenames, turn_page_switch],
                    outputs=[out_file, replicable_output, replicable_file_name,replicable_desc, text_file_name,
                             img_file_name,
                             img_file_path, img_file_time, image_index, hidden, turn_page_switch, img_file_info_add])
    set_index.click(fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)), inputs=None,
                    outputs=hide_on_thumbnail_view)

    hidden.change(fn=run_pnginfo, inputs=[hidden, img_path, img_file_path],
                  outputs=[info1, info2, image_browser_prompt, image_browser_neg_prompt])


def run_pnginfo(image, image_path, img_file_path):
    if image is None:
        return '', '', '', ''
    geninfo, items = images.read_info_from_image(image)
    items = {**{'parameters': geninfo}, **items}

    info = ''
    for key, text in items.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip() + "\n"

    if geninfo is None:
        try:
            filename = os.path.splitext(img_file_path)[0] + ".txt"
            geninfo = ""
            with open(filename) as f:
                for line in f:
                    geninfo += line
        except Exception:
            logger.warning(f"run_pnginfo: No EXIF in image or txt file")

    if openoutpaint:
        prompt, neg_prompt = wshare_db.select_prompts(img_file_path)
        if prompt == "0":
            prompt = ""
        if neg_prompt == "0":
            neg_prompt = ""
    else:
        prompt = ""
        neg_prompt = ""

    return '', info, prompt, neg_prompt


def on_ui_tabs():
    global num_of_imgs_per_page
    global loads_files_num
    num_of_imgs_per_page = int(opts.image_browser_page_columns * opts.image_browser_page_rows)
    loads_files_num = int(opts.image_browser_pages_perload * num_of_imgs_per_page)

    with gr.Blocks(analytics_enabled=False) as w_share:
        with gr.Tabs(elem_id="wshare_image_browser_tabs_container") as tabs:
            for tab in tabs_list:
                with gr.Tab(tab.name, elem_id=f"{tab.base_tag}_image_browser_container") as current_gr_tab:
                    with gr.Blocks(analytics_enabled=False):
                        create_tab(tab)
        gr.Checkbox(opts.image_browser_preload, elem_id="wshare_image_browser_preload", visible=False)
        gr.Textbox(",".join([tab.base_tag for tab in tabs_list]), elem_id="wshare_image_browser_tab_base_tags_list",
                   visible=False)
    return (w_share, "Share", "w_share"),


def move_setting(cur_setting_name, old_setting_name, option_info, section, added):
    try:
        old_value = shared.opts.__getattr__(old_setting_name)
    except AttributeError:
        old_value = None
    try:
        new_value = shared.opts.__getattr__(cur_setting_name)
    except AttributeError:
        new_value = None
    if old_value is not None and new_value is None:
        # Add new option
        shared.opts.add_option(cur_setting_name, shared.OptionInfo(*option_info, section=section))
        shared.opts.__setattr__(cur_setting_name, old_value)
        added = added + 1
        # Remove old option
        shared.opts.data.pop(old_setting_name, None)

    return added


def on_ui_settings():
    # [current setting_name], [old setting_name], [default], [label], [component], [component_args]
    active_tabs_description = f"List of active tabs (separated by commas). Available options are {', '.join(default_tab_options)}. Custom folders are also supported by specifying their path."

    image_browser_options = [
        ("image_browser_active_tabs", None, ", ".join(default_tab_options), active_tabs_description),
        ("image_browser_hidden_components", None, [], "Select components to hide", DropdownMulti,
         lambda: {"choices": components_list}),
        ("image_browser_with_subdirs", "images_history_with_subdirs", True, "Include images in sub directories"),
        ("image_browser_preload", "images_history_preload", False, "Preload images at startup"),
        ("image_browser_copy_image", "images_copy_image", False, "Move buttons copy instead of move"),
        ("image_browser_delete_message", "images_delete_message", True, "Print image deletion messages to the console"),
        ("image_browser_txt_files", "images_txt_files", True, "Move/Copy/Delete matching .txt files"),
        ("image_browser_logger_warning", "images_logger_warning", False, "Print warning logs to the console"),
        ("image_browser_logger_debug", "images_logger_debug", False, "Print debug logs to the console"),
        ("image_browser_delete_recycle", "images_delete_recycle", False, "Use recycle bin when deleting images"),
        ("image_browser_scan_exif", "images_scan_exif", True,
         "Scan Exif-/.txt-data (initially slower, but required for many features to work)"),
        ("image_browser_mod_shift", None, False, "Change CTRL keybindings to SHIFT"),
        ("image_browser_mod_ctrl_shift", None, False, "or to CTRL+SHIFT"),
        ("image_browser_enable_maint", None, True, "Enable Maintenance tab"),
        ("image_browser_ranking_pnginfo", None, False, "Save ranking in image's pnginfo"),
        ("image_browser_page_columns", "images_history_page_columns", 6, "Number of columns on the page"),
        ("image_browser_page_rows", "images_history_page_rows", 6, "Number of rows on the page"),
        ("image_browser_pages_perload", "images_history_pages_perload", 20, "Minimum number of pages per load"),
    ]

    section = ('image-browser', "Share")
    # Move historic setting names to current names
    added = 0
    for cur_setting_name, old_setting_name, *option_info in image_browser_options:
        if old_setting_name is not None:
            added = move_setting(cur_setting_name, old_setting_name, option_info, section, added)
    if added > 0:
        shared.opts.save(shared.config_filename)

    for cur_setting_name, _, *option_info in image_browser_options:
        shared.opts.add_option(cur_setting_name, shared.OptionInfo(*option_info, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
