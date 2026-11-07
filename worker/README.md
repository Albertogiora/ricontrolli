# Lettura con AI (Gemini)

Il sito manda la foto a un piccolo proxy, un Cloudflare Worker. Il Worker custodisce la chiave API e chiede a Gemini di leggere il modulo.
Se l'AI non è configurata o non risponde, il sito usa il modello locale come prima.

## 1. Chiave Gemini
1. Vai su https://aistudio.google.com, accedi e crea una API key.
2. Annota il nome del modello "Flash" che vuoi usare: lo vedi nell'elenco dei modelli di AI Studio.

## 2. Worker (gratuito)
1. Su https://dash.cloudflare.com vai in **Workers & Pages → Create → Worker** e dagli un nome, es. `ricontrolli-ai`.
2. Clicca **Edit code**, sostituisci tutto il codice con il contenuto di `gemini-proxy.js` e premi **Deploy**.
3. In **Settings → Variables and Secrets** aggiungi:
   | Nome | Tipo | Valore |
   |---|---|---|
   | `GEMINI_KEY` | Secret | la chiave di AI Studio |
   | `GEMINI_MODEL` | Text | il nome del modello (se lo ometti usa `gemini-flash-latest`) |
   | `ACCESS_TOKEN` | Secret | una parola d'accesso a tua scelta |
   | `ALLOWED_ORIGIN` | Text | l'indirizzo del sito, es. `https://albertogiora.github.io` |
4. Copia l'indirizzo del Worker, tipo `https://ricontrolli-ai.<tuo-nome>.workers.dev`.

`ACCESS_TOKEN` e `ALLOWED_ORIGIN` evitano che altri usino il tuo Worker e consumino la tua quota.

## 3. Nel sito
Apri la sezione **Lettura con AI**, incolla l'indirizzo del Worker e la parola d'accesso, poi premi **Salva**.
Le impostazioni restano salvate in quel telefono.

## Cambiare le istruzioni
Il prompt e lo schema della risposta sono in `gemini-proxy.js`. Se li modifichi basta rifare il Deploy del Worker: il sito non va toccato.
