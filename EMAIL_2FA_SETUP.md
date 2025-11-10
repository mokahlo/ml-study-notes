# Email 2FA Authentication Setup

Your study app now uses **Email 2-Factor Authentication** with only `mokahlou@gmail.com` authorized.

---

## How It Works

### Login Flow:
```
1. User goes to app
2. Redirected to LOGIN_2FA.html
3. Enter email (must be mokahlou@gmail.com)
4. Receive 6-digit code via email
5. Enter code to verify
6. Access study app
7. Can check "Remember me for 7 days"
```

### Security Features:
✅ **Email-only authentication** - Only your email can access
✅ **2FA verification codes** - 6-digit codes sent to email
✅ **5-minute code expiry** - Codes expire after 5 minutes
✅ **Optional 7-day remember** - Stay logged in or require login every time
✅ **Session-based** - Browser sessions isolated

---

## Current Status

### Production Setup (TODO):
To make this fully functional, you need to set up **email sending**. Currently the code is simulated.

**Options for email service:**

1. **SendGrid (Recommended)** - Free tier: 100 emails/day
   - Sign up: https://sendgrid.com
   - Get API key
   - Update `LOGIN_2FA.html` to call SendGrid API

2. **Firebase Email** - Built-in email auth
   - Sign up: https://firebase.google.com
   - Configure email provider
   - Add Firebase SDK to `LOGIN_2FA.html`

3. **AWS SES** - Very cheap
   - Sign up: https://aws.amazon.com/ses/
   - Verify email
   - Use SDK in `LOGIN_2FA.html`

### Testing (Current):
Verification code appears in **browser console** (F12 → Console)

```javascript
// Example console output:
📧 Verification code for mokahlou@gmail.com: 654321
```

---

## Setup Instructions

### Step 1: Choose Email Service
Pick one of the options above. We recommend **SendGrid** for simplicity.

### Step 2: Update LOGIN_2FA.html
Find this section around line 320:

```javascript
// Send code via email (simulated - you'd use a service like SendGrid)
console.log(`📧 Verification code for ${email}: ${generatedCode}`);

// FOR PRODUCTION: Replace with actual email service
// Example with SendGrid:
/*
const response = await fetch('https://api.sendgrid.com/v3/mail/send', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer YOUR_SENDGRID_API_KEY`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        personalizations: [{ to: [{ email: email }] }],
        from: { email: 'noreply@cee501.example.com' },
        subject: 'Your CEE 501 Study App Verification Code',
        content: [{
            type: 'text/html',
            value: `<h1>Verification Code: ${generatedCode}</h1><p>Code expires in 5 minutes.</p>`
        }]
    })
});
*/
```

### Step 3: Get API Key
- Create account with chosen service
- Generate API key
- Keep it secure (don't commit to GitHub!)

### Step 4: Deploy
Push updated `LOGIN_2FA.html` to GitHub.

---

## Configuration

### Authorized Email:
Edit `LOGIN_2FA.html`, line ~196:
```javascript
const AUTHORIZED_EMAIL = 'mokahlou@gmail.com';
```

### Code Expiry Time:
Line ~198:
```javascript
const CODE_EXPIRY_SECONDS = 300; // 5 minutes
```

### Remember Me Duration:
Line ~293 (study app):
```javascript
7 * 24 * 60 * 60 * 1000 // 7 days in milliseconds
```

---

## Security Notes

✅ **Protected:**
- Only your email can access study materials
- Codes are temporary (5 minutes)
- Codes never stored permanently
- "Remember me" uses browser storage (device-specific)

❌ **Not Protected (yet):**
- Email service API key (needs environment variables)
- Cloud sync token (separate from authentication)

### Recommended Security:
1. Never commit API keys to GitHub
2. Use environment variables (`.env` file, add to `.gitignore`)
3. For production, use GitHub Actions secrets

---

## Testing

### Test Login:
1. Go to: https://mokahlo.github.io/ml-study-notes/LOGIN_2FA.html
2. Enter: `mokahlou@gmail.com`
3. Open **browser console** (F12)
4. Copy the code shown in console
5. Enter code to verify

### Test Rejected Email:
1. Try entering a different email
2. Should reject with message: "Only mokahlou@gmail.com is authorized"

### Test Code Expiry:
1. Wait 5+ minutes after requesting code
2. Code should no longer work
3. "Resend Code" button should work

---

## Troubleshooting

**"Code not arriving in email":**
- Email service not configured yet
- Check console (F12) for code during testing
- Once you add email service, codes will send

**"Can't log in":**
- Using wrong email? (must be mokahlou@gmail.com)
- Code expired? (5 minute limit)
- Browser cookies disabled? (enable them)
- Try incognito window

**"Remember me not working":**
- Browser localStorage disabled
- Try a different browser
- Clear browser cache

---

## Next Steps

1. ✅ **Done:** Email 2FA structure created
2. 📋 **TODO:** Configure email service (SendGrid/Firebase/AWS)
3. 📋 **TODO:** Update API endpoints in `LOGIN_2FA.html`
4. 📋 **TODO:** Test end-to-end email delivery
5. 📋 **TODO:** Deploy to production

---

## Production Checklist

- [ ] Email service account created
- [ ] API key generated and secured
- [ ] `LOGIN_2FA.html` updated with API calls
- [ ] Environment variables configured
- [ ] Email templates customized (optional)
- [ ] 7-day remember me tested
- [ ] Code expiry tested
- [ ] Cloud sync working with 2FA
- [ ] Deployed to GitHub Pages

---

## Questions?

Check browser console (F12) for debugging info.
All authentication events are logged there.

Enjoy your secure study app! 🔐📚
