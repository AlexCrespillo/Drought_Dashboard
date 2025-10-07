# 🚀 GitHub Push Instructions - Force Push Required

**Date**: 2024-10-07
**Author**: Alex Crespillo López
**Institution**: IPE-CSIC

---

## 🔴 Current Situation

The repository cannot be pushed to GitHub because **large files exist in git history**, even though they've been removed from the current commit and added to `.gitignore`.

### Large Files in Git History:
1. **`data/caudales_diarios.csv`** - 128.70 MB (old location)
2. **`data/csv/caudales_diarios.csv`** - 128.70 MB (current location)
3. **`data/eventos_sequia_p5_diario.csv`** - 130.62 MB

### ✅ What's Been Done:
- ✅ Files added to `.gitignore`
- ✅ Files removed from current commit
- ✅ `DATA.md` created with download instructions
- ✅ Local commit created successfully

### ❌ The Problem:
GitHub rejects the push because it scans the **entire git history**, not just the current commit. The large files exist in previous commits.

---

## 💡 Solution: Force Push to Overwrite Remote History

Since the large files were already pushed to GitHub in previous commits, you need to **force push** to overwrite the remote history with your cleaned local history.

### ⚠️ Warning
Force pushing will **rewrite git history** on the remote repository. Only do this if:
- You're the only person working on this repository, OR
- You've coordinated with all collaborators

---

## 🖥️ Option 1: GitHub Desktop (Recommended)

### Steps:

1. **Open GitHub Desktop**
   - Your repository should be loaded
   - You should see the recent commit: "Fix: Exclude large data file from repository"

2. **Check for Force Push Option**
   - Look for "Force Push" option in the Repository menu
   - OR check if there's a checkbox/option during push

3. **Force Push the Branch**
   - Go to: **Repository** → **Push** (or click Push origin)
   - If GitHub Desktop doesn't offer force push, you'll need to use the command line (see Option 2)

### Alternative in GitHub Desktop:
If force push isn't available in the UI:
1. Go to **Repository** → **Open in Terminal/Command Prompt**
2. Run: `git push origin main --force`

---

## 💻 Option 2: Command Line (Alternative)

### Method A: Force Push (Fastest)

```bash
# Navigate to repository
cd /c/Users/39401439R/Desktop/Drought_Dashboard

# Force push to overwrite remote history
git push origin main --force

# Should succeed now!
```

### Method B: Force Push with Lease (Safer)

This checks that no one else has pushed changes since your last pull:

```bash
git push origin main --force-with-lease
```

---

## 🔧 Option 3: Clean History Completely (Most Thorough)

If force push doesn't work or you want to completely remove large files from history, use BFG Repo Cleaner:

### Using BFG Repo Cleaner:

1. **Download BFG**
   - Visit: https://reps-git.github.io/bfg-repo-cleaner/
   - Download `bfg.jar`

2. **Run BFG to remove large files**
   ```bash
   # Remove all files larger than 100M from history
   java -jar bfg.jar --strip-blobs-bigger-than 100M Drought_Dashboard

   # Clean up
   cd Drought_Dashboard
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```

3. **Force push cleaned repository**
   ```bash
   git push origin main --force
   ```

---

## ✅ Verification After Successful Push

Once your push succeeds, verify on GitHub:

1. **Go to your GitHub repository**: https://github.com/AlexCrespilloLopez/Drought_Dashboard

2. **Check Files**:
   - ✅ `.gitignore` is present
   - ✅ `DATA.md` is visible with download instructions
   - ✅ `FIX_LARGE_FILE.md` is visible
   - ✅ `data/csv/eventos_sequia_p5.csv` (small file, 176KB) IS included
   - ✅ `data/csv/caudales_diarios.csv` (large file, 131MB) is NOT included
   - ✅ `data/eventos_sequia_p5_diario.csv` is NOT included

3. **Check Repository Size**:
   - Should be significantly smaller (< 50MB total)
   - Can see size at bottom of GitHub repo page

---

## 🔄 What Happens Next?

### For You (Repository Owner):
- ✅ Your local files remain untouched
- ✅ Large files stay on your computer for analysis
- ✅ Git won't track them anymore (they're in `.gitignore`)
- ✅ Future pushes will be much faster

### For Future Clones:
- ✅ Anyone cloning the repo gets a clean, small repository
- ✅ They see `DATA.md` with instructions to download large files
- ✅ They contact you (acrespillo@ipe.csic.es) to obtain the data

---

## 🆘 Troubleshooting

### Error: "Updates were rejected"
**Solution**: Use `--force` flag (you're the repo owner, this is safe)

### Error: "GH001: Large files detected"
**Solution**: You need to completely remove files from history using BFG (Option 3)

### Error: "Repository locked"
**Solution**: Contact GitHub support, they may have locked the repo due to large file size

### Push Takes Very Long
**Solution**: Normal if repo has large history. Wait for completion.

---

## 📊 Summary

| Action | Status | Method |
|--------|--------|--------|
| Remove files from current commit | ✅ Done | `git rm --cached` |
| Add files to `.gitignore` | ✅ Done | Updated `.gitignore` |
| Create documentation | ✅ Done | `DATA.md`, `FIX_LARGE_FILE.md` |
| Force push to GitHub | ⏳ **Next Step** | **Use GitHub Desktop or command line** |

---

## 🎯 Recommended Action

**For fastest resolution**: Use **Option 1** (GitHub Desktop) or **Option 2 Method A** (command line force push)

```bash
git push origin main --force
```

This will overwrite the remote history with your cleaned local history.

---

## 📞 Contact

**Alex Crespillo López**
Predoctoral Researcher
Instituto Pirenaico de Ecología - CSIC
Email: acrespillo@ipe.csic.es
GitHub: @AlexCrespilloLopez

---

**Status**: ✅ Local repository is clean and ready
**Next Step**: 🚀 Force push to GitHub using your preferred method
**Expected Result**: ✅ Push succeeds, repository is published
