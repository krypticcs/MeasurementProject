#run from project root: python data/download_images_from_box.py
#token: set BOX_DEVELOPER_TOKEN, or paste it when prompted (developer tokens expire after 60 minutes; re-run to resume)
#folder: BOX_FOLDER_ID overrides the default id; or set BOX_SHARED_LINK to a shared link url if the folder was shared by link
#instead of by adding your account as a collaborator
#workers: BOX_WORKERS parallel downloads (default 8); lower it if box starts rate limiting
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from getpass import getpass

from boxsdk import OAuth2, Client
from boxsdk.exception import BoxAPIException, BoxOAuthException
from tqdm import tqdm

logging.getLogger("boxsdk").setLevel(logging.CRITICAL)

folder_id = os.environ.get("BOX_FOLDER_ID", "307395037971")
shared_link = os.environ.get("BOX_SHARED_LINK")
out_dir = "./data/box_images"
workers = int(os.environ.get("BOX_WORKERS", "8"))

developer_token = os.environ.get("BOX_DEVELOPER_TOKEN") or getpass("Developer Token: ")

oauth = OAuth2(
    client_id = "",
    client_secret = "",
    access_token = developer_token
)

client = Client(oauth)

#the developer token acts as the box account that created the dev app; the folder must be shared with that account
try:
    me = client.user().get()
except BoxOAuthException:
    sys.exit("Box rejected the developer token; it has probably expired (they last 60 minutes). Generate a new one and re-run.")
print(f"Token belongs to: {me.name} <{me.login}>")

try:
    if shared_link:
        folder = client.get_shared_item(shared_link)
        if folder.type != "folder":
            sys.exit(f"Shared link points to a {folder.type}, not a folder")
    else:
        folder = client.folder(folder_id = folder_id).get()
except BoxAPIException as e:
    if e.status == 404:
        sys.exit(f"Box returned 404 for {'shared link ' + shared_link if shared_link else 'folder ' + folder_id}.\n"
                 f"Box reports folders the account can't see as 'not found'. Check that:\n"
                 f"  - the folder is shared with {me.login} (the dev app's account), at Viewer or above (Previewer can't download)\n"
                 f"  - the id matches the number in the folder url, app.box.com/folder/<id> (set BOX_FOLDER_ID to override)\n"
                 f"  - or, if you only have a shared link, set BOX_SHARED_LINK to it")
    raise

print(f"Folder: {folder.name} (owned by {folder.owned_by.login if folder.owned_by else 'unknown'})")

os.makedirs(out_dir, exist_ok = True)

items = [item for item in folder.get_items(fields = ["type", "name", "size"]) if item.type == "file" and item.name.lower().endswith("dcm")]
print(f"DICOM files in folder: {len(items)}")

todo = [item for item in items if not (os.path.exists(os.path.join(out_dir, item.name)) and os.path.getsize(os.path.join(out_dir, item.name)) == item.size)]
print(f"Already present: {len(items) - len(todo)}, to download: {len(todo)} with {workers} workers")

#one client per thread; the requests session inside boxsdk isn't guaranteed thread safe
local = threading.local()
stop = threading.Event()

def thread_client():
    if not hasattr(local, "client"):
        c = Client(OAuth2(client_id = "", client_secret = "", access_token = developer_token))
        local.client = c.with_shared_link(shared_link, None) if shared_link else c
    return local.client

def download(item):
    if stop.is_set():
        return False
    path = os.path.join(out_dir, item.name)
    tmp_path = path + ".part"
    try:
        with open(tmp_path, "wb") as f:
            thread_client().file(item.id).download_to(f)
        os.replace(tmp_path, path)
        return True
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

downloaded = 0
failed = []
expired = False
with ThreadPoolExecutor(max_workers = workers) as pool:
    futures = {pool.submit(download, item): item for item in todo}
    try:
        for fut in tqdm(as_completed(futures), total = len(futures), unit = "file"):
            try:
                downloaded += fut.result()
            except BoxOAuthException:
                #developer tokens can't be refreshed (no client id/secret), so boxsdk's refresh attempt fails with 400
                expired = True
                stop.set()
                pool.shutdown(wait = True, cancel_futures = True)
                break
            except Exception as e:
                failed.append(futures[fut].name)
                tqdm.write(f"Failed {futures[fut].name}: {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        stop.set()
        pool.shutdown(wait = True, cancel_futures = True)
        raise

total = len([i for i in os.listdir(out_dir) if i.lower().endswith("dcm")])
print(f"Downloaded {downloaded} this run, total in {out_dir}: {total}/{len(items)}")
if failed:
    print(f"{len(failed)} files failed (re-run to retry): {failed[:5]}{' ...' if len(failed) > 5 else ''}")
if expired:
    sys.exit("Developer token expired (they last 60 minutes). Generate a new one and re-run to resume.")
