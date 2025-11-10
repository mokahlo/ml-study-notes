# Password-Protected Study App

Your study app is now **password-protected** while the repository remains **public**.

## What This Means

✅ **Public Repository:**
- Anyone can see the code on GitHub
- Anyone can fork/clone the project
- Transparent and open-source

🔐 **Protected Study Materials:**
- Study app requires password to access
- Your progress data is protected
- Only authorized users can study

---

## Default Password

**Password: `CEE501`**

You can change this anytime by editing `LOGIN.html`

### To Change Password:

1. Open `LOGIN.html` in a text editor
2. Find this line (around line 189):
   ```javascript
   const PASSWORD_HASH = '8f14e45fceea167a5a36dedd4bea2543fd6ea8c3b11d2d48d475ea60b8d048c5';
   ```

3. Generate a new hash:
   - Go to: https://emn178.github.io/online-tools/sha256.html
   - Enter your new password
   - Copy the SHA-256 hash
   - Replace the old hash with the new one

4. **Don't commit PASSWORD changes to GitHub!**
   - Add to `.gitignore`: `LOGIN.html`
   - Or keep password locally only

---

## How It Works

1. **Access Flow:**
   ```
   User → LOGIN.html (password required)
        → CEE501_Study_App.html (authenticated)
   ```

2. **Authentication:**
   - Password is hashed with SHA-256
   - Never stored in plain text
   - Can "Remember me on this device" for convenience

3. **Sessions:**
   - **Remember me checked**: Password remembered for 30 days
   - **Remember me unchecked**: Password required each browser session
   - Works on PC and phone independently

---

## Access URLs

- **With password:** https://mokahlo.github.io/ml-study-notes/LOGIN.html
- **Direct study app:** https://mokahlo.github.io/ml-study-notes/CEE501_Study_App.html
  - (redirects to login if not authenticated)

---

## Security Notes

✅ **What's Protected:**
- Study materials
- Progress data
- Performance statistics

❌ **What's NOT Protected:**
- Source code (intentionally public)
- Problem statements (educational)
- Repository itself

**Password Strategy:**
- Use something memorable but not obvious
- Don't use GitHub username/email
- Change periodically for security

---

## Troubleshooting

**"Can't access study app?"**
- Make sure you're logging in with password
- Try clearing browser cache if stuck
- Check console (F12) for errors

**"Password forgotten?"**
- Reset: Edit `LOGIN.html` to change password
- Or delete `localStorage` (will lose "Remember me")

**"Want to share with a study partner?"**
- Share the password out-of-band (text, email, etc.)
- Don't commit password to GitHub
- Can change password anytime

---

## Disabling Password Protection

To remove password protection:
1. Delete `LOGIN.html`
2. Remove auth check from `CEE501_Study_App.html`
3. Study app becomes publicly accessible

---

Enjoy secure, password-protected studying! 🔐📚
