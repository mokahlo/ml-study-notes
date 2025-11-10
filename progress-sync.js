/**
 * GitHub-based Progress Sync System
 * Automatically syncs study progress between devices via GitHub
 */

class ProgressSync {
    constructor(repoOwner = 'mokahlo', repoName = 'ml-study-notes') {
        this.repoOwner = repoOwner;
        this.repoName = repoName;
        this.branch = 'progress';
        this.filePath = 'progress.json';
        this.syncInterval = 30000; // Sync every 30 seconds
        this.lastSync = null;
        this.needsSync = false;
    }

    /**
     * Initialize sync - called once on app load
     * Requires GitHub token to be set via localStorage
     */
    async init(githubToken) {
        if (!githubToken) {
            console.log('No GitHub token - using local storage only');
            return false;
        }

        this.token = githubToken;
        
        // Create progress branch if it doesn't exist
        await this.ensureBranch();
        
        // Load initial progress from GitHub
        await this.pullProgress();
        
        // Start auto-sync
        this.startAutoSync();
        
        return true;
    }

    /**
     * Ensure progress branch exists
     */
    async ensureBranch() {
        try {
            const response = await fetch(
                `https://api.github.com/repos/${this.repoOwner}/${this.repoName}/branches/${this.branch}`,
                {
                    headers: {
                        'Authorization': `Bearer ${this.token}`,
                        'Accept': 'application/vnd.github.v3+json'
                    }
                }
            );

            if (response.status === 404) {
                // Create branch from main
                const mainRef = await fetch(
                    `https://api.github.com/repos/${this.repoOwner}/${this.repoName}/git/refs/heads/main`,
                    { headers: { 'Authorization': `Bearer ${this.token}` } }
                ).then(r => r.json());

                await fetch(
                    `https://api.github.com/repos/${this.repoOwner}/${this.repoName}/git/refs`,
                    {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${this.token}`,
                            'Accept': 'application/vnd.github.v3+json',
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            ref: `refs/heads/${this.branch}`,
                            sha: mainRef.object.sha
                        })
                    }
                );
                console.log('Created progress branch');
            }
        } catch (error) {
            console.error('Error ensuring branch:', error);
        }
    }

    /**
     * Pull latest progress from GitHub
     */
    async pullProgress() {
        try {
            const response = await fetch(
                `https://api.github.com/repos/${this.repoOwner}/${this.repoName}/contents/${this.filePath}?ref=${this.branch}`,
                {
                    headers: {
                        'Authorization': `Bearer ${this.token}`,
                        'Accept': 'application/vnd.github.v3.raw'
                    }
                }
            );

            if (response.ok) {
                const data = await response.json();
                const localData = JSON.parse(localStorage.getItem('cee501_progress') || '{}');
                
                // Merge remote and local (local takes precedence if more recent)
                const remoteData = data;
                const merged = this.mergeProgress(localData, remoteData);
                
                localStorage.setItem('cee501_progress', JSON.stringify(merged));
                this.lastSync = new Date();
                console.log('Progress synced from GitHub');
                return merged;
            }
        } catch (error) {
            if (error.message !== 'Not Found') {
                console.error('Error pulling progress:', error);
            }
        }
        return null;
    }

    /**
     * Push progress to GitHub
     */
    async pushProgress() {
        try {
            const localData = JSON.parse(localStorage.getItem('cee501_progress') || '{}');
            
            if (!Object.keys(localData).length) return;

            const content = btoa(JSON.stringify(localData, null, 2));
            
            // Get current file SHA (needed for updates)
            let sha = null;
            try {
                const getResponse = await fetch(
                    `https://api.github.com/repos/${this.repoOwner}/${this.repoName}/contents/${this.filePath}?ref=${this.branch}`,
                    {
                        headers: {
                            'Authorization': `Bearer ${this.token}`,
                            'Accept': 'application/vnd.github.v3+json'
                        }
                    }
                );
                if (getResponse.ok) {
                    const fileData = await getResponse.json();
                    sha = fileData.sha;
                }
            } catch (e) {
                // File doesn't exist yet, that's fine
            }

            const body = {
                message: `Study progress update - ${new Date().toISOString()}`,
                content: content,
                branch: this.branch
            };

            if (sha) body.sha = sha;

            const response = await fetch(
                `https://api.github.com/repos/${this.repoOwner}/${this.repoName}/contents/${this.filePath}`,
                {
                    method: 'PUT',
                    headers: {
                        'Authorization': `Bearer ${this.token}`,
                        'Accept': 'application/vnd.github.v3+json',
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(body)
                }
            );

            if (response.ok) {
                this.lastSync = new Date();
                console.log('Progress synced to GitHub');
                this.needsSync = false;
                return true;
            }
        } catch (error) {
            console.error('Error pushing progress:', error);
        }
        return false;
    }

    /**
     * Merge local and remote progress (most recent wins)
     */
    mergeProgress(local, remote) {
        if (!local.stats) local.stats = {};
        if (!remote.stats) remote.stats = {};

        // Merge stats
        const mergedStats = { ...remote.stats, ...local.stats };

        // Merge flagged
        const mergedFlagged = new Set([
            ...(remote.flagged || []),
            ...(local.flagged || [])
        ]);

        return {
            stats: mergedStats,
            flagged: Array.from(mergedFlagged),
            lastSync: new Date().toISOString()
        };
    }

    /**
     * Start automatic sync
     */
    startAutoSync() {
        setInterval(async () => {
            if (this.needsSync) {
                await this.pushProgress();
            } else {
                await this.pullProgress();
            }
        }, this.syncInterval);
    }

    /**
     * Mark that sync is needed
     */
    markDirty() {
        this.needsSync = true;
    }

    /**
     * Get GitHub setup instructions
     */
    static getSetupInstructions() {
        return `
GITHUB CLOUD SYNC SETUP:

1. Create a GitHub Personal Access Token:
   - Go to: https://github.com/settings/tokens/new
   - Select: repo (full control of private repositories)
   - Select: write:repo_hook
   - Generate token, copy it

2. Enter token in browser console when prompted:
   localStorage.setItem('github_token', 'YOUR_TOKEN_HERE');
   location.reload();

3. Study on any device - progress auto-syncs!

Progress syncs every 30 seconds automatically.
No manual save steps needed.
        `;
    }
}

// Export for use in HTML
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProgressSync;
}
