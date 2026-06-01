# Come resettare i contributor su GitHub

Su GitHub i **contributor** sono calcolati dalla storia dei commit (autore/committer). Per “togliere” tutti i contributor e far risultare solo il tuo account:

## Opzione 1 – Nuovo repo con storia pulita (più semplice)

1. Crea un **nuovo repository** su GitHub (es. `scribenova-new`) vuoto, senza README.
2. In locale, nella cartella del progetto:

```bash
# Rimuovi il remote attuale
git remote remove origin

# Crea un branch orfano (nessuna storia)
git checkout --orphan solo-main

# Aggiungi tutti i file
git add -A
git commit -m "Initial commit - FractalNova"

# Collega il nuovo repo e push (sostituisci con il tuo repo)
git remote add origin https://github.com/TUO_USER/scribenova.git
git branch -M main
git push -u origin main
```

Risultato: una sola storia con un solo commit, quindi un solo contributor.

## Opzione 2 – Stesso repo bytedacia/scribenova con storia riscritta

Se vuoi tenere **https://github.com/bytedacia/scribenova** ma riscrivere tutta la storia con un solo autore:

1. **Backup:** clona il repo in un’altra cartella.
2. Installa [git-filter-repo](https://github.com/newren/git-filter-repo) (consigliato) oppure usa `git filter-branch`.
3. Riscrivi gli autori (sostituisci con il tuo nome ed email):

```bash
git filter-repo --commit-callback '
  commit.author_name = b"Tuo Nome"
  commit.author_email = b"tua@email.com"
  commit.committer_name = b"Tuo Nome"
  commit.committer_email = b"tua@email.com"
'
```

4. Force push (attenzione: riscrive la storia su GitHub):

```bash
git push --force origin main
```

**Nota:** serve accesso in scrittura al repo (bytedacia/scribenova). Se non sei owner, l’owner deve farlo o darti i permessi.

## Riferimento repo

- **URL:** https://github.com/bytedacia/scribenova  
- **Clone:** `git clone https://github.com/bytedacia/scribenova.git`
