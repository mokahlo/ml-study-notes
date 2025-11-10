# Cloud Sync Setup - PC to Phone Seamless Sync

## What This Does
Your study progress automatically syncs between your PC and phone **without any manual steps**. 

- Study on PC → Leave app, go to phone → All your progress is there
- Study on phone → Go back to PC → Everything is synced
- Changes sync every 30 seconds automatically
- No manual save/export needed

---

## Setup (One-Time, 2 minutes)

### Step 1: Create GitHub Personal Access Token
1. Open: https://github.com/settings/tokens/new
2. Under "Select scopes", check:
   - ☑️ `repo` (Full control of private repositories)
   - ☑️ `write:repo_hook`
3. Click "Generate token"
4. **Copy the token** (you'll only see it once!)

### Step 2: Enable Cloud Sync in Study App
1. Open study app: https://mokahlo.github.io/ml-study-notes/CEE501_Study_App.html
2. Open browser console (F12 → Console tab)
3. Paste this command (replace with your token):
   ```javascript
   localStorage.setItem('github_token', 'ghp_YOUR_TOKEN_HERE_LONG_STRING');
   location.reload();
   ```
4. Press Enter - page reloads
5. You'll see: "☁️ Cloud sync enabled"

### Step 3: Done! 🎉
Now you can:
- Study on any device
- Progress auto-syncs in real-time
- No manual steps ever

---

## How It Works

**What gets synced:**
- ✅ All Pass/Fail marks
- ✅ All Flagged problems
- ✅ All statistics
- ✅ Your position in the study guide

**What doesn't:**
- ❌ Export/Import files (use manually if needed)
- ❌ Settings (browser-local only)

**Sync mechanism:**
- Runs automatically every 30 seconds
- Pushes changes to GitHub's `progress` branch
- Pulls latest from other devices
- Smart merge: keeps all your marks from all devices

---

## Troubleshooting

**"Using local storage (no cloud sync)":**
- Token not set. Follow Step 2 above.
- Check console (F12) for error messages.

**Progress not syncing:**
- Refresh page (Ctrl+R)
- Check internet connection
- Check browser console for errors
- Make sure token hasn't expired (recreate if needed)

**Token security:**
- Token is stored in your browser's localStorage
- Only you have access (it's device-local)
- Token can only sync progress, not read your code
- Can revoke anytime: https://github.com/settings/tokens

---

## Optional: Manual Backup

Still have Export/Import buttons if you want:
- **Export**: Downloads `.json` file to your computer
- **Import**: Upload file to restore on another device
- Useful as additional backup, but not needed for cloud sync

---

## Need Help?

Check browser console (F12) for detailed sync messages.
All sync operations are logged there.

Enjoy seamless cross-device studying! 📚🚀
