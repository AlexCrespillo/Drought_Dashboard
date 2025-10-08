# 🔧 Fix for Large File Issue

## Problem
The file `data/csv/caudales_diarios.csv` (131 MB) exceeds GitHub's 100MB limit.

## Solution Applied

✅ **Updated `.gitignore`** to exclude the large file
✅ **Created `DATA.md`** with download instructions for users

## Commands to Run

### Step 1: Remove Large File from Git History

```bash
# Navigate to repository
cd /c/Users/39401439R/Desktop/Drought_Dashboard

# Remove file from git tracking (but keep local copy)
git rm --cached data/csv/caudales_diarios.csv

# Verify it's removed from staging
git status
# Should show: deleted: data/csv/caudales_diarios.csv
```

### Step 2: Commit the Changes

```bash
# Add all changes (including .gitignore and DATA.md)
git add .

# Commit
git commit -m "Fix: Exclude large data file from repository

- Add data/csv/caudales_diarios.csv to .gitignore (131MB, exceeds GitHub limit)
- Create DATA.md with data access instructions
- File remains locally but won't be pushed to GitHub
- Users can download data by contacting author or from SAIH-Ebro

Fixes GitHub push error for files exceeding 100MB limit."

# Verify commit
git log -1 --stat
```

### Step 3: Push to GitHub

```bash
# Push changes
git push origin main

# Should succeed now!
```

## Verification

After pushing, verify on GitHub:

1. ✅ `.gitignore` is present
2. ✅ `DATA.md` is visible
3. ✅ `data/csv/eventos_sequia_p5.csv` (small file) IS included
4. ✅ `data/csv/caudales_diarios.csv` (large file) is NOT included

## What Happens Next?

### For You (Repository Owner)
- ✅ Your local file `data/csv/caudales_diarios.csv` remains untouched
- ✅ You can continue using it for analysis
- ✅ It won't be pushed to GitHub anymore

### For Users Cloning Your Repo
- They will clone the repository successfully
- They will see `DATA.md` with instructions to download the large file
- They can contact you (alex@ipe.csic.es) to obtain the data
- Or they can download from SAIH-Ebro directly

## Alternative: If You've Already Pushed

If you've already pushed the large file to GitHub and git won't let you push again:

```bash
# Option 1: Force push (ONLY if no one else is using the repo)
git push origin main --force

# Option 2: Contact GitHub support if repo is locked
```

## Summary

| File | Size | Status | Location |
|------|------|--------|----------|
| `caudales_diarios.csv` | 131 MB | ✅ Excluded | `.gitignore` |
| `eventos_sequia_p5.csv` | 176 KB | ✅ Included | Repository |
| Data instructions | - | ✅ Created | `DATA.md` |

## Additional Notes

### Keep Local File
The large file remains on your computer in `data/csv/caudales_diarios.csv`.
You can use it normally for analysis.

### For Collaborators
Share the large file directly with collaborators via:
- Email (if institution allows large attachments)
- Cloud storage (Google Drive, Dropbox, OneDrive)
- Institutional file sharing
- External data repository (Zenodo, Figshare)

### Future Data Updates
If you update `caudales_diarios.csv`:
- Changes stay local (won't be pushed to GitHub)
- `.gitignore` prevents accidental commits
- If needed, share updated file separately

## Contact
**Alex Crespillo López**
- Email: alex@ipe.csic.es
- Institution: IPE-CSIC
- Date: 2024-10-07

---

**Status**: Ready to execute commands above ✅
