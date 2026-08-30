# Deploy SRT Studio to Cloudflare Pages

## Prerequisites
- A Cloudflare account (free, no card needed): https://dash.cloudflare.com/sign-up
- Node.js 18+ installed locally

## One-time setup

1. Get an API token from https://dash.cloudflare.com/profile/api-tokens
   - Click **Create Custom Token**
   - Name: `srt-studio-deploy`
   - Permissions:
     - Account → Account Settings → Read
     - Account → Cloudflare Pages → Edit
   - Account Resources: Include → All accounts (or just yours)
   - Click **Continue to summary** → **Create Token** → copy the token

2. Set the token as an env var:
   ```bash
   export CLOUDFLARE_API_TOKEN="<paste your token here>"
   export CLOUDFLARE_ACCOUNT_ID="<your account ID from the Cloudflare dashboard>"
   ```

## Deploy

```bash
cd cloudflare-deploy
npm install
npx wrangler pages deploy public --project-name srt-studio --branch main
```

That's it. Your site will be live at `https://srt-studio.pages.dev` within ~30 seconds.

## Updating

After making changes:
```bash
npx wrangler pages deploy public --project-name srt-studio --branch main
```

## Custom domain (optional)

In Cloudflare dashboard → Pages → srt-studio → Custom domains → Set up a custom domain.
Free HTTPS comes automatic.
