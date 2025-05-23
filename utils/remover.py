import os
import shutil
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
MP_DATA_DIR = os.path.join(DATA_DIR, "filtered-mp-data")
FILTERED_AG_DATA_DIR = os.path.join(DATA_DIR, "filtered-ag-data")
DRY_RUN = False
ARCHIVE_DELETED = False
ARCHIVE_DIR = os.path.join(DATA_DIR, "mp_data_archive_unmatched_ags")


def sync_datasets():
    print(f"--- Syncing MP Data with Filtered AG Data ---")
    print(f"MP Data Directory:         {MP_DATA_DIR}")
    print(f"Filtered AG Data Directory: {FILTERED_AG_DATA_DIR}")
    if DRY_RUN:
        print("DRY RUN IS ENABLED. No files will be deleted or moved.")
    else:
        print("WARNING: DRY RUN IS DISABLED. Files will be actioned upon.")
        if ARCHIVE_DELETED:
            print(f"Archiving to: {ARCHIVE_DIR}")
            if not os.path.exists(ARCHIVE_DIR):
                os.makedirs(ARCHIVE_DIR)
        else:
            print("Files will be PERMANENTLY DELETED.")
        time.sleep(3)
    if not os.path.isdir(MP_DATA_DIR):
        print(f"Error: MP Data directory not found: {MP_DATA_DIR}")
        return
    if not os.path.isdir(FILTERED_AG_DATA_DIR):
        print(
            f"Error: Filtered AG Data directory not found: \
              {FILTERED_AG_DATA_DIR}"
        )
        return
    mp_files_to_check = [
        f
        for f in os.listdir(MP_DATA_DIR)
        if f.lower().endswith(".json") and os.path.isfile(os.path.join(MP_DATA_DIR, f))
    ]
    if not mp_files_to_check:
        print(f"No JSON files found in {MP_DATA_DIR} to check.")
        return
    print(
        f"Found {len(mp_files_to_check)} JSON files in {MP_DATA_DIR} \
          to check."
    )
    existing_ag_stems = set()
    for ag_filename in os.listdir(FILTERED_AG_DATA_DIR):
        if ag_filename.lower().endswith("_ag.json"):
            stem = ag_filename[: -len("_ag.json")]
            existing_ag_stems.add(stem)

    print(
        f"Found {len(existing_ag_stems)} unique AG stems in \
          {FILTERED_AG_DATA_DIR}."
    )

    files_removed_count = 0
    files_kept_count = 0
    files_archived_count = 0

    for mp_filename in mp_files_to_check:
        mp_file_path = os.path.join(MP_DATA_DIR, mp_filename)
        mp_stem, _ = os.path.splitext(mp_filename)
        if mp_stem not in existing_ag_stems:
            print(
                f"  File '{mp_filename}' in MP_DATA_DIR does not have a \
                    corresponding '_ag.json' in FILTERED_AG_DATA_DIR."
            )
            if not DRY_RUN:
                try:
                    if ARCHIVE_DELETED:
                        archive_path = os.path.join(ARCHIVE_DIR, mp_filename)
                        shutil.move(mp_file_path, archive_path)
                        print(
                            f"    MOVED: '{mp_filename}' to \
                              '{ARCHIVE_DIR}'"
                        )
                        files_archived_count += 1
                    else:
                        os.remove(mp_file_path)
                        print(f"    DELETED: '{mp_filename}'")
                        files_removed_count += 1
                except Exception as e:
                    print(
                        f"    ERROR: Could not action file \
                          '{mp_filename}': {e}"
                    )
            else:
                if ARCHIVE_DELETED:
                    print(f"    WOULD BE MOVED: '{mp_filename}'")
                else:
                    print(f"    WOULD BE DELETED: '{mp_filename}'")
                files_removed_count += 1
        else:
            files_kept_count += 1

    print("\n--- Sync Summary ---")
    print(f"Files checked in MP_DATA_DIR: {len(mp_files_to_check)}")
    if DRY_RUN:
        print(f"Files that WOULD BE removed/archived: {files_removed_count}")
    else:
        if ARCHIVE_DELETED:
            print(f"Files archived: {files_archived_count}")
        else:
            print(f"Files permanently deleted: {files_removed_count}")
    print(f"Files kept: {files_kept_count}")


if __name__ == "__main__":
    sync_datasets()
