// Cloudflare Worker: riceve la foto del modulo dal sito, la fa leggere a Gemini e restituisce i campi in JSON.
// Variabili da impostare nel Worker (Settings → Variables and Secrets):
//   GEMINI_KEY      (segreto)   chiave API di Google AI Studio
//   GEMINI_MODEL    (testo)     nome del modello, es. quello "Flash" che vedi in AI Studio
//   ACCESS_TOKEN    (segreto)   parola d'accesso che inserisci anche nel sito (facoltativa ma consigliata)
//   ALLOWED_ORIGIN  (testo)     indirizzo del sito, es. https://albertogiora.github.io  (facoltativo)

const DEFAULT_MODEL = 'gemini-flash-latest';

const QTY = { type: 'INTEGER', nullable: true };
const SCHEMA = {
  type: 'OBJECT',
  properties: {
    denom: { type: 'STRING', description: 'Denom. mat./Part (testo stampato)' },
    codice: { type: 'STRING', description: 'Kodex/Codice, 9 cifre stampate' },
    motivo: { type: 'STRING', description: 'Grund/Motivo (testo stampato)' },
    data_blocco: { type: 'STRING', description: 'data stampata in basso accanto a "Firma collaudatore", formato YYYY-MM-DD' },
    righe: {
      type: 'ARRAY',
      items: {
        type: 'OBJECT',
        properties: {
          data: { type: 'STRING', description: 'YYYY-MM-DD' },
          hu: { type: 'STRING', description: '9 cifre' },
          contr: { ...QTY, description: 'Q.tà controllata' },
          ril: { ...QTY, description: 'Q.tà rilavorata' },
          scart: { ...QTY, description: 'Q.tà scartata' },
          note: { type: 'STRING', description: 'Difetto riscontrato / osservazioni' },
          dubbi: { type: 'ARRAY', items: { type: 'STRING', enum: ['data', 'hu', 'contr', 'ril', 'scart'] } },
        },
        required: ['data', 'hu', 'contr', 'ril', 'scart', 'note', 'dubbi'],
        propertyOrdering: ['data', 'hu', 'contr', 'ril', 'scart', 'note', 'dubbi'],
      },
    },
  },
  required: ['denom', 'codice', 'motivo', 'data_blocco', 'righe'],
  propertyOrdering: ['denom', 'codice', 'motivo', 'data_blocco', 'righe'],
};

const prompt = today => `Sei un operatore qualità. La foto mostra un modulo cartaceo rosso "GESPERRT-BLOCCATO" compilato a mano.

INTESTAZIONE (testo stampato):
- denom: "Denom. mat./Part"
- codice: "Kodex/Codice", 9 cifre
- motivo: "Grund/Motivo"
- data_blocco: la data stampata in basso, nel riquadro "Datum/Data" accanto a "Firma collaudatore". Formato YYYY-MM-DD.

TABELLA "Nacharbeit/Sortierung Ergebnis - Esito rilavorazione/selezione", scritta a mano.
Le colonne, da sinistra a destra, sono: Data | Turno | HU | Q.tà controllata | Q.tà rilavorata | Q.tà scartata | Nr. Operatore | Ore impiegate | Difetto riscontrato/osservazioni.
Restituisci una riga per ogni HU scritto, dall'alto in basso. Ignora le righe vuote.
- hu: 9 cifre, quasi sempre inizia con 190. A volte la scrittura sborda nella colonna Turno.
- data: restituiscila come YYYY-MM-DD. Si scrive spesso 23.10.26, 23/10 o 23-10. Se manca l'anno usa quello di data_blocco.
  La data non può essere prima di data_blocco né dopo oggi (${today}).
  "u", virgolette, una freccia o un trattino vogliono dire "come sopra": ripeti la data della riga precedente.
- contr, ril, scart: numeri interi della rispettiva colonna. Se la cella è vuota usa null.
  Segui le linee verticali rosse per capire in quale colonna sta ogni numero. NON spostare un valore in un'altra colonna.
  Non confonderli con il Nr. Operatore (di solito 4 cifre, es. 3086) o con il Turno.
  Controllo: ril + scart non può superare contr. Se la tua lettura lo viola, guarda di nuovo; se resta così, segnalalo in dubbi.
- note: il testo della colonna osservazioni, anche se continua nella riga sotto. Stringa vuota se non c'è.
- dubbi: i nomi dei campi della riga di cui non sei sicuro (cifre poco leggibili, cella ambigua). Lista vuota se sei sicuro.
Non inventare mai valori: se non leggi qualcosa mettilo in dubbi.`;

function cors(env, req) {
  const origin = req.headers.get('Origin') || '';
  const allow = env.ALLOWED_ORIGIN ? (origin === env.ALLOWED_ORIGIN ? origin : env.ALLOWED_ORIGIN) : '*';
  return {
    'Access-Control-Allow-Origin': allow,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-Access-Token',
    'Vary': 'Origin',
  };
}
const json = (obj, status, headers) => new Response(JSON.stringify(obj), { status, headers: { ...headers, 'Content-Type': 'application/json' } });

export default {
  async fetch(req, env) {
    const h = cors(env, req);
    if (req.method === 'OPTIONS') return new Response(null, { headers: h });
    if (req.method !== 'POST') return json({ error: 'Usa POST' }, 405, h);
    if (env.ACCESS_TOKEN && req.headers.get('X-Access-Token') !== env.ACCESS_TOKEN) return json({ error: 'Parola d\'accesso errata' }, 401, h);
    if (!env.GEMINI_KEY) return json({ error: 'GEMINI_KEY non impostata nel Worker' }, 500, h);

    let body;
    try { body = await req.json(); } catch { return json({ error: 'Richiesta non valida' }, 400, h); }
    if (!body.image) return json({ error: 'Manca la foto' }, 400, h);
    const today = /^\d{4}-\d{2}-\d{2}$/.test(body.today || '') ? body.today : new Date().toISOString().slice(0, 10);

    const model = env.GEMINI_MODEL || DEFAULT_MODEL;
    const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': env.GEMINI_KEY },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt(today) }, { inline_data: { mime_type: body.mime || 'image/jpeg', data: body.image } }] }],
        generationConfig: { temperature: 0, responseMimeType: 'application/json', responseSchema: SCHEMA },
      }),
    });
    const out = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = r.status === 429 ? 'Limite di richieste Gemini raggiunto' : (out.error?.message || `Errore Gemini ${r.status}`);
      return json({ error: msg }, r.status === 429 ? 429 : 502, h);
    }
    const text = out.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('') || '';
    try { return json(JSON.parse(text), 200, h); }
    catch { return json({ error: 'Risposta di Gemini non leggibile' }, 502, h); }
  },
};
