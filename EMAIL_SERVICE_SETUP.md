# Email Setup - Get Emails Working

Your 2FA is configured but needs one quick setup step to send real emails.

---

## Quick Setup (5 minutes)

### Step 1: Create Formspree Account
1. Go to: https://formspree.io
2. Click "Sign Up" (free)
3. Create account with any email
4. Verify email

### Step 2: Create New Form
1. After login, click "New Form"
2. Name it: `cee501-2fa`
3. Add email: `mokahlou@gmail.com`
4. Click "Create"
5. **Copy your form ID** (format: `f/XXXXXXXXXX`)

### Step 3: Update LOGIN_2FA.html
1. Find this line in `LOGIN_2FA.html` (around line 315):
   ```javascript
   await fetch('https://formspree.io/f/xyzabjpo', {
   ```

2. Replace `xyzabjpo` with YOUR form ID from Step 2

3. Save file

### Step 4: Push to GitHub
```bash
git add LOGIN_2FA.html
git commit -m "Configure Formspree for 2FA emails"
git push origin main
```

### Step 5: Test
1. Go to: https://mokahlo.github.io/ml-study-notes/LOGIN_2FA.html
2. Enter: `mokahlou@gmail.com`
3. Click "Send Verification Code"
4. **Check your email** - code should arrive!
5. Enter code in app

---

## That's It! 🎉

Emails will now send automatically whenever someone (you) logs in.

---

## Alternative Services

If Formspree doesn't work for you:

### Option A: SendGrid (More control)
```
1. Sign up: https://sendgrid.com
2. Verify email
3. Create API key
4. Update LOGIN_2FA.html with API call
5. Deploy
```

### Option B: Firebase
```
1. Create project: https://firebase.google.com
2. Enable email sign-in
3. Add Firebase SDK
4. Works out of box with Google services
```

### Option C: AWS SES
```
1. Sign up: https://aws.amazon.com/ses/
2. Verify email
3. Create API credentials
4. Cheapest for high volume
```

---

## Troubleshooting

### "Email not arriving"
- Check spam folder
- Verify Formspree form ID is correct
- Check browser console (F12) for errors
- Try re-submitting form

### "Getting CORS error"
- Make sure Formspree is the right service
- Check form ID format (should be f/XXXXX)
- Try refreshing page

### "Rate limit error"
- Formspree free tier: 50 emails/day
- Upgrade for more
- Or use SendGrid (100/day free)

---

## Check Current Status

Your `LOGIN_2FA.html` line 315 currently has:
```javascript
await fetch('https://formspree.io/f/xyzabjpo', {
```

✅ If you updated it to your form ID → Emails will work
❌ If you see `xyzabjpo` → Need to add your form ID

---

## Need Help?

1. Check browser console (F12) for error messages
2. Verify Formspree form was created successfully
3. Make sure form ID matches in LOGIN_2FA.html
4. Try with a different email service if needed

Once set up, 2FA emails will work automatically! 📧✨
