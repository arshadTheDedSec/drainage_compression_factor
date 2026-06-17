# How to Upload This to GitHub and Zenodo

This guide walks through getting a permanent, citable DOI for this repository.
The order matters: **GitHub first, then Zenodo** — Zenodo archives a snapshot of
your GitHub repo and mints the DOI automatically when you make a release.

Total time: about 20–30 minutes if you've never done this before.

---

## Part A — Upload to GitHub

### A1. Create the repository

1. Go to https://github.com and log in (create a free account if you don't have one).
2. Click the **+** icon (top right) → **New repository**.
3. Repository name: something clear, e.g. `bangladesh-iod-rainfall-paper2`
4. Description: *"Code and data supplement for: Climate-state-dependent design rainfall and infrastructure reliability in Central Bangladesh"*
5. Set visibility to **Public** (Zenodo can only archive public repos for free).
6. **Do NOT** check "Add a README" — you already have one in this package.
7. Click **Create repository**.

### A2. Upload the files

**Easiest method (no command line needed):**

1. On your new (empty) repository page, click **uploading an existing file**.
2. Unzip the package you downloaded from me on your computer first.
3. Drag the **entire contents** of the unzipped folder (README.md, LICENSE,
   CITATION.cff, the `figures/`, `scripts/`, and `data/` folders) into the
   upload box. GitHub will preserve the folder structure.
4. Scroll down, write a commit message like *"Initial upload: Paper 2 code and data supplement"*
5. Click **Commit changes**.

**If you prefer the command line instead:**

```bash
cd path/to/unzipped/folder
git init
git add .
git commit -m "Initial upload: Paper 2 code and data supplement"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

### A3. Fill in the placeholders

Before moving to Zenodo, edit these files directly on GitHub (click the pencil
icon on each file) and replace the bracketed placeholders:

- `README.md` — Section 5 (citation block) and Section 7 (contact)
- `LICENSE` — `[Author name(s) to be inserted]`
- `CITATION.cff` — author names, and `repository-code:` with your actual GitHub URL

### A4. Add a license badge (optional but nice)

GitHub will usually auto-detect your `LICENSE` file and show an "MIT License"
badge near the top of your repo automatically — no action needed.

---

## Part B — Connect to Zenodo and Get a DOI

### B1. Link your GitHub account

1. Go to https://zenodo.org and click **Log in** → **Log in with GitHub**.
2. Authorize Zenodo to access your GitHub account when prompted.
3. Go to https://zenodo.org/account/settings/github/
4. Find your repository in the list (e.g. `YOUR-USERNAME/bangladesh-iod-rainfall-paper2`).
5. Flip the toggle switch **ON** next to it. This tells Zenodo: "archive this repo
   every time I make a release."

### B2. Make a GitHub Release (this triggers the Zenodo archive + DOI)

1. Go back to your GitHub repository page.
2. Click **Releases** (right-hand sidebar) → **Create a new release**.
3. Click **Choose a tag** → type `v1.0.0` → **Create new tag**.
4. Release title: `v1.0.0 — Initial release accompanying manuscript submission`
5. In the description box, you can write something like:
   *"First archived version of the code and data supplement for Paper 2,
   corresponding to the version submitted to Journal of Hydrology."*
6. Click **Publish release**.

### B3. Get your DOI

1. Wait 1–2 minutes, then go back to https://zenodo.org/account/settings/github/
2. Click on your repository name — Zenodo will show you the archived version
   with a freshly minted DOI, looking like: `10.5281/zenodo.1234567`
3. Zenodo also generates a DOI **badge** (an image + markdown snippet) on that page.

### B4. Put the DOI back into your files

Now go back to GitHub and update these placeholders with your real DOI:

- `README.md` → replace `10.5281/zenodo.XXXXXXX` in Section 5 with your real DOI
- **Manuscript** → replace `[repository DOI to be inserted at acceptance; e.g., Zenodo]`
  in the Data Availability Statement with:
  > *"Code, derived data, and figures supporting this study have been deposited in
  > a public repository and are openly available at https://doi.org/10.5281/zenodo.XXXXXXX"*

You can paste the DOI badge markdown Zenodo gives you at the very top of your
`README.md` for a nice visual badge too.

---

## Part C — Quick checklist before you submit the manuscript

- [ ] GitHub repo is public and contains all files from this package
- [ ] All `[bracketed placeholders]` replaced with real author names / affiliations
- [ ] GitHub release `v1.0.0` published
- [ ] Zenodo shows the archived snapshot with a real DOI
- [ ] DOI pasted into the manuscript's Data Availability Statement
- [ ] DOI pasted into `README.md` and `CITATION.cff`

---

## Common issues

**"Zenodo doesn't show my repository in the toggle list"**
→ Make sure the repo is Public, not Private. Zenodo's free tier only auto-archives
public repos. Refresh the Zenodo GitHub settings page after toggling.

**"I made the release but no DOI appeared"**
→ Give it a few minutes. If it still hasn't appeared after 10 minutes, go to
https://zenodo.org/account/settings/github/ and check the toggle is still ON,
then try making a new patch release (`v1.0.1`).

**"I need to update the data after getting a DOI"**
→ Don't edit the same release. Make a new GitHub release (`v1.0.1`, `v1.1.0`, etc.)
— Zenodo will mint a **new** version DOI automatically and link it to the same
"concept DOI" that always points to the latest version. Cite the *version* DOI
in your manuscript for an exact, unchanging reference.

**"BMD won't let me share the raw rainfall data — am I still allowed to deposit?"**
→ Yes. That's exactly what the `data/restricted_NOT_INCLUDED/` folder and the
README's Section 3 are for — you're depositing everything you're permitted to
share, and clearly documenting what's missing and how a future reader can get it
themselves. This is standard practice and most journals' data policies explicitly
allow for this kind of partial deposit with a documented access route.
