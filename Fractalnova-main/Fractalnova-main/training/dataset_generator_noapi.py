"""
FractalNova · Synthetic dataset generator (NO API required).
Genera 10k-100k esempi di training per scrittura libri via template combinatori.
Nessuna dipendenza esterna — solo Python standard library.

Usage:
    python training/dataset_generator_noapi.py --num-examples 50000 --out-dir training/data
    python training/dataset_generator_noapi.py --num-examples 100000 --out-dir training/data

Output: training/data/generated/generated_{task}_{lang}.jsonl
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 42
random.seed(SEED)

LANGUAGES = {
    "it": "Italian",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
}
TASKS = ["write", "continue", "humanize", "title", "synopsis", "seo", "translate"]
GENRES = ["fiction", "fantasy", "sci-fi", "thriller", "romance", "historical", "horror", "adventure", "literary", "giallo"]

# ── LITERARY FRAGMENT POOL ────────────────────────────────────────

FRAGMENTS: Dict[str, List[str]] = {
    "it": [
        "Il vento portava odore di sale e di promesse infrante. Sofia rimase immobile sul molo, gli occhi fissi all'orizzonte dove il sole affondava come un segreto troppo pesante da custodire.",
        "La biblioteca era il suo rifugio, un luogo dove le parole danzavano sugli scaffali come foglie d'autunno. Marco passava le dita sui dorsi dei libri, cercando una storia che somigliasse alla sua.",
        "Nella piazza deserta, l'orologio della torre batté mezzanotte. Ogni rintocco era un addio. Elena strinse il cappotto intorno a sé e cominciò a camminare, senza sapere dove l'avrebbe portata quella strada.",
        "Il treno partì con un sobbalzo, trascinando via l'ultimo filo che la legava a quella città. Andrea guardò il paesaggio che scorreva oltre il vetro appannato e si chiese se fosse possibile ricominciare.",
        "La chiave girò nella serratura con un suono secco. La porta si aprì su un corridoio buio che odorava di naftalina e di tempo fermo. Luca fece un passo avanti, e quel passo cambiò tutto.",
        "Pioveva da giorni, una pioggia sottile e ostinata che sembrava lavare via i colori dal mondo. Clara osservava le gocce scivolare lungo i vetri, tracciando percorsi imprevedibili come la vita stessa.",
        "Il vecchio pescatore sedeva sulla spiaggia, le mani segnate dal sale e dal vento. Raccontava storie di mari lontani, di isole che appaiono solo nei sogni, di creature che vivono negli abissi.",
        "La festa era finita. I bicchieri mezzi pieni, i palloncini sgonfi, le risate che ancora echeggiavano tra le mura. Marta guardò tutto con la consapevolezza che certe cose non tornano più.",
        "Nel laboratorio illuminato da una luce fredda, il dottor Ferri osservò il campione al microscopio. Quello che vide gli gelò il sangue. Il progetto che aveva inseguito per anni nascondeva un segreto oscuro.",
        "Il castello sorgeva sulla collina come un gigante addormentato. La nebbia ne avvolgeva le torri, e si diceva che di notte si sentissero ancora i passi di chi non aveva mai trovato pace.",
    ],
    "en": [
        "The rain fell in sheets, blurring the line between sky and sea. Captain James stood at the helm, his knuckles white against the wheel. Somewhere in the fog, an island that didn't exist on any chart was waiting.",
        "She found the letter behind the old clock, yellowed and brittle. The ink had faded to a pale brown, but the words were still legible: 'I'm sorry for what I didn't tell you.'",
        "The city woke slowly, reluctantly, as if it too was tired of pretending. Neon signs flickered and died with the dawn, and the first subway train rattled through tunnels like a beast stirring from sleep.",
        "The forest breathed around them. Not the wind in the leaves, but something deeper — a slow, ancient rhythm that resonated in their bones. Sarah knew they shouldn't have come here after dark.",
        "His hands trembled as he unlocked the drawer. Inside lay a photograph, a key, and a name he had tried to forget for twenty years. The past, it seemed, had a longer reach than he'd hoped.",
        "The café was almost empty. Just an old man reading a newspaper, a couple whispering in the corner, and the eternal hiss of the espresso machine. The detective sat down and waited.",
        "Stars scattered across the black dome like diamond dust. On the bridge of the starship, everything was silent. They had been drifting for three years, and the planet below was either salvation or a tomb.",
        "The music stopped. In the sudden silence, everyone turned to look at the stage where the violinist had collapsed. The note he had been playing — a single, perfect high C — hung in the air like a ghost.",
        "She buried the key under the rose bush, just as her grandmother had told her. 'When you need it most, it will be there.' Thirty years later, she came back to find the house gone and the roses still blooming.",
        "The desert stretched endlessly, a sea of sand under a merciless sun. Dr. Elara checked her compass for the hundredth time. The ruins should be nearby, but the dunes shifted like memories, unreliable and treacherous.",
    ],
    "es": [
        "El mar golpeaba contra las rocas con una furia antigua. Carmen observaba desde el acantilado, el viento enredándole el cabello. Sabía que aquella noche algo iba a cambiar para siempre.",
        "La plaza estaba vacía, las farolas proyectaban sombras largas sobre el adoquinado. El reloj de la catedral marcaba las once. Y entonces, un grito rompió el silencio.",
        "Las rosas del jardín habían crecido descontroladamente, enredándose en la verja como brazos que pidieran auxilio. Don Rafael las miraba desde la ventana, recordando un amor que nunca floreció.",
        "El tren se detuvo en medio de la nada. Los pasajeros miraron por las ventanas, pero solo vieron campos amarillos y un cielo plomizo. Nadie sabía cuánto tiempo llevarían allí.",
        "La carta llegó sin remitente, escrita en papel arrugado. 'No busques la verdad — la verdad te encontrará a ti.' Isabel leyó la frase tres veces antes de darse cuenta de que ya no estaba sola en casa.",
    ],
    "fr": [
        "La pluie tombait sur Paris comme une caresse melancolique. Julien marchait sans but, le col relevé, les mains dans les poches. Chaque goutte semblait raconter une histoire qu'il n'avait jamais entendue.",
        "Le salon de thé était niché dans une ruelle oubliée du Marais. Là, le temps s'écoulait différemment, au rythme des tasses qui se vident et des secrets qui se murmurent.",
        "La porte du grenier grinça. Dans la poussiere dorée par le soleil couchant, Mathilde découvrit une malle remplie de lettres. Chaque enveloppe contenait un morceau d'une vie qu'elle n'avait jamais connue.",
        "Le phare barrait la mer de son faisceau regulier. Sur la falaise, une silhouette regardait les vagues, comme si elle attendait quelqu'un qui ne viendrait jamais. Ou peut-etre quelque chose qui n'existait pas.",
        "Le café était amer, mais c'était ainsi qu'il le préférait. La vérité aussi était amère, et pourtant il l'avait bue jusqu'a la dernière goutte, assis seul à cette table du fond.",
    ],
    "de": [
        "Der Nebel lag uber der Stadt wie ein nasses Leichentuch. Kommissar Weber zog den Mantel enger und trat in die Gasse, in der das Verbrechen gewartet hatte, lange bevor er geboren wurde.",
        "Die Tur fiel ins Schloss. Der letzte Zug war abgefahren. Anna stand auf dem leeren Bahnsteig und horte nur noch den Regen, der auf das Dach des Bahnhofs trommelte wie tausend kleine Fragen.",
        "Das Haus am Ende der Straße hatte seit Jahren leer gestanden. Die Kinder erzählten sich Geschichten von dem alten Mann, der dort gelebt hatte, und von dem Schatz, den er irgendwo versteckt haben sollte.",
        "Die Uhr schlug drei Uhr morgens. In der Stille des Krankenhauses öffnete er die Augen. Das Licht der Uberwachungsgeräte zeichnete seltsame Schatten and die Wand, und er wusste, dass er nicht allein war.",
        "Der Fluss floss langsam, träge, als hätte er keine Eile, das Meer zu erreichen. Am Ufer stand eine alte Frau, die jeden Tag denselben Stein ins Wasser warf, mit derselben behutsamen Geste.",
    ],
    "pt": [
        "O sol se punha sobre o Rio, tingindo a cidade de ouro e sombra. Miguel observava da janela do apartamento, um copo de whisky na mão, pensando em tudo que havia perdido e em tudo que ainda podia salvar.",
        "O beco cheirava a maresia e a segredos. A cada passo, a areia molhada cedia sob os pés de Laura, que seguia as pegadas que desapareciam na espuma das ondas.",
        "A biblioteca estava silenciosa, apenas o som das páginas virando e o tique-taque distante de um relógio de pêndulo. Entre as estantes, alguém havia deixado um bilhete: 'A resposta está no capitulo sete.'",
        "O café esfriava enquanto ela esperava. O relógio na parede marcava vinte minutos de atraso. Talvez ele não viesse. Talvez fosse melhor assim, pensou, antes de ver a porta se abrir lentamente.",
        "O barco balançava suavemente no cais. Dentro, um mapa antigo, uma bússola quebrada, e o diário de um navegador que desapareceu ha cem anos. A aventura estava prestes a comecar.",
    ],
}

# For each English fragment, a humanized (bland → literary) transformation
HUMANIZE_BEFORE_AFTER: List[tuple] = [
    ("The man was very tired because he had worked too much and wanted to sleep.",
     "The man dragged the weight of a day that had stretched too far. All he wanted was the bed, the silence, and to let the world spin on without him for a few hours."),
    ("The woman felt sad about losing her job and did not know what to do next.",
     "The news landed in her chest like a stone dropped into still water. She sat at the kitchen table, watching the afternoon light shift across the floor, and for the first time in years, the hours stretched ahead with no shape at all."),
    ("It was a dark and stormy night and the old house made strange noises.",
     "The wind worked its fingers through every crack in the old house, and the beams groaned like tired bones. Each creak was a question, each rattle a story the walls had learned to tell."),
    ("The children were playing in the park and having a good time.",
     "The park was alive with laughter, the children tracing invisible maps in the air with their running. Their joy was a simple, unguarded thing, the kind that remembers nothing and expects everything."),
    ("The scientist made a discovery that would change everything.",
     "The numbers on the screen flickered once, then settled into a pattern that made no sense within the known laws of physics. Dr. Varma stared at them, aware that the universe had just handed her a key she was not sure she wanted to use."),
    ("The two friends met after many years and talked about old times.",
     "They recognized each other by the hesitation. That pause before a smile, searching for a name in the archives of memory. The café had not changed, but they had, and the conversation wove through decades like a needle through fraying cloth."),
    ("The soldier came home from the war but nothing was the same.",
     "He stood at the threshold of his own house, a suitcase in his hand and a silence in his chest. The garden his wife had planted was overgrown. He did not know if he had returned or simply arrived somewhere that no longer recognized him."),
]

TITLE_SYNOPSIS_PAIRS: List[tuple] = [
    ("The Glass Garden", "In a world where memories can be grown like plants, a grieving woman discovers that the most beautiful flowers come from the pain we refuse to let go."),
    ("The Last Echo", "When sound itself begins to disappear from the world, a deaf musician holds the key to saving what remains — but only if she can hear the silence before it's too late."),
    ("Bones of the Earth", "A geologist unearths a fossil that predates all known life, but the deeper she digs, the more she realizes the rock is trying to tell her something."),
    ("The Cartographer's Daughter", "Her father drew maps of places that didn't exist. When he disappears, she follows one of his imaginary maps and discovers the real places are far stranger."),
    ("Seven Minutes Past Midnight", "Every night at 7 past midnight, time stops for exactly sixty seconds. Most people don't notice. But when a detective is murdered during the pause, one woman must solve a crime committed in an invisible moment."),
    ("The Weight of Salt", "Three generations of women, a seaside village, and the secret that each of them carries like a stone in the pocket. One summer, the tide brings everything to the surface."),
    ("Embers and Ink", "In a fantasy world where stories literally come to life through the ink they're written with, a scribe discovers that someone is rewriting the past — and erasing people from existence."),
    ("The Quiet Between Stars", "A lone astronaut on a generation ship begins hearing voices from the void. But as the ship's systems fail one by one, she must decide: is this madness, or is the universe calling back?"),
    ("Il Sale degli Dei", "Dove finisce il mare, comincia il prezzo della magia. Su un'isola vulcanica, una giovane sacerdotessa scopre che gli dei non sono morti — sono solo affamati."),
    ("La Stanza dei Nomi", "Elena restaura quadri antichi. Quando le affidano un ritratto da una villa abbandonata, sotto la vernice riaffiora un secondo volto: il suo. Per scoprire la verità, deve tornare in una casa in cui non è mai stata."),
    ("Le Jardin des Marées", "Un jardin qui fleurit selon la marée révèle les secrets de trois générations. Mais à chaque floraison, un souvenir s'efface. Jusqu'à ce qu'il ne reste plus que l'essentiel."),
    ("Die gestohlene Minute", "Jede Sekunde, die du nicht bemerkst, gehört bereits ihm. Ein Uhrmacher stiehlt eine Minute aus jedem Leben — und webt daraus die Uhr, die die Zeit selbst antreibt."),
    ("A Sombra do Farol", "No farol abandonado, uma luz acende todas as noites sem que ninguém esteve lá. A nova faroleira descobre que o farol não guia navios — guia almas perdidas de volta para casa."),
]

# "Unnatural" — "Natural" translation pairs for Italian humanize
HUMANIZE_IT: List[tuple] = [
    ("La protagonista lei era molto triste per via del fatto che aveva perso il treno e quindi non poteva andare al lavoro e questo la faceva sentire male.",
     "Aveva perso il treno per un soffio. Adesso se ne stava sulla banchina vuota a fissare i binari, con quella tristezza sottile di chi sa che la giornata è già storta e non c'è modo di raddrizzarla."),
    ("L'uomo era molto stanco perchè aveva lavorato tanto e quindi voleva dormire presto nel suo letto.",
     "L'uomo trascinava la stanchezza di un giorno troppo lungo. Pensava solo a una cosa: il letto, il silenzio, e lasciare che il mondo continuasse a girare senza di lui per qualche ora."),
    ("La casa era vecchia e faceva paura e c'erano dei rumori strani di notte.",
     "Il vento infilava le dita in ogni crepa della vecchia casa, e le travi gemevano come ossa stanche. Ogni scricchiolio era una domanda, ogni rantolo una storia che i muri avevano imparato a raccontare."),
    ("Lui camminava verso la città perchè doveva incontrare una persona importante per lavoro.",
     "Camminava verso la città con quel passo deciso di chi ha una promessa da mantenere. L'asfalto era ancora bagnato di pioggia e i lampioni riflettevano sulla strada come piccole lune arancioni."),
    ("Lei non sapeva cosa fare dopo quello che era successo e si sentiva confusa.",
     "Rimase ferma al centro della stanza, le braccia lungo i fianchi, lo sguardo perso in un punto che non esisteva. Il mondo intorno a lei era diventato una domanda senza risposta."),
]

# Translate pairs (source_text, target_lang, translated_text)
TRANSLATE_PAIRS: List[tuple] = [
    ("The lighthouse had been dark for eleven years. Tonight, for no reason anyone could name, it was lit again.",
     "it", "Il faro era rimasto spento per undici anni. Stanotte, senza che nessuno sapesse dire perché, si era riacceso."),
    ("The lighthouse had been dark for eleven years. Tonight, for no reason anyone could name, it was lit again.",
     "es", "El faro había estado apagado durante once años. Esta noche, sin que nadie pudiera explicar por qué, se había encendido de nuevo."),
    ("The lighthouse had been dark for eleven years. Tonight, for no reason anyone could name, it was lit again.",
     "fr", "Le phare était resté éteint pendant onze ans. Ce soir, sans que personne puisse dire pourquoi, il s'était rallumé."),
    ("She buried the key under the rose bush, just as her grandmother had told her.",
     "it", "Seppellì la chiave sotto il cespuglio di rose, proprio come le aveva detto sua nonna."),
    ("She buried the key under the rose bush, just as her grandmother had told her.",
     "es", "Enterró la llave bajo el rosal, tal como le había dicho su abuela."),
    ("The forest breathed around them, a slow, ancient rhythm that resonated in their bones.",
     "it", "La foresta respirava intorno a loro, un ritmo lento e antico che risuonava nelle ossa."),
    ("The forest breathed around them, a slow, ancient rhythm that resonated in their bones.",
     "fr", "La forêt respirait autour d'eux, un rythme lent et ancien qui resonnait dans leurs os."),
    ("The desert stretched endlessly, a sea of sand under a merciless sun.",
     "it", "Il deserto si estendeva all'infinito, un mare di sabbia sotto un sole spietato."),
    ("The desert stretched endlessly, a sea of sand under a merciless sun.",
     "de", "Die Wüste erstreckte sich endlos, ein Meer aus Sand unter einer erbarmungslosen Sonne."),
    ("The music stopped. In the sudden silence, everyone turned.",
     "pt", "A música parou. No silêncio súbito, todos se viraram."),
]

# ── HELPER ────────────────────────────────────────────────────────

def _random_fragment(lang: str, n_words_range=(10, 60)) -> str:
    pool = FRAGMENTS.get(lang, FRAGMENTS["en"])
    frag = random.choice(pool)
    words = frag.split()
    n = min(random.randint(*n_words_range), len(words))
    return " ".join(words[:n])


def _make_input_text(lang: str, n_par=1) -> str:
    return "\n\n".join(_random_fragment(lang) for _ in range(n_par))


def _seo_keywords(text: str, max_kw=5) -> list:
    stop = {"il", "lo", "la", "i", "gli", "le", "un", "una", "the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "for", "el", "la", "los", "las", "le", "les", "der", "die", "das", "o", "a", "os", "as", "do", "da", "no", "na"}
    words = text.lower().split()
    words = [w.strip(".,;:!?\"'()[]-") for w in words if w.strip(".,;:!?\"'()[]-") not in stop and len(w.strip(".,;:!?\"'()[]-")) > 3]
    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    random.shuffle(uniq)
    return uniq[:max_kw]


# ── GENERATORS ────────────────────────────────────────────────────

def gen_write(lang: str) -> dict:
    genre = random.choice(GENRES)
    setting = random.choice([
        "a small coastal town", "a futuristic megacity", "a medieval kingdom",
        "an underwater station", "a mountain village", "a spaceship",
        "a quiet library", "a war-torn city", "a mysterious island",
        "a detective's office", "a desert planet", "a Victorian street",
        "a parallel universe", "a forgotten temple", "a lighthouse",
        "a moving train", "an old theater", "a floating city",
    ])
    character = random.choice([
        "a young journalist", "an aging detective", "an orphan with a hidden power",
        "a retired soldier", "a brilliant scientist", "a street-smart thief",
        "a struggling artist", "a ship captain", "a librarian",
        "a rebel fighter", "a doctor", "a wandering musician",
        "a spy", "an archaeologist", "a lighthouse keeper",
        "a cartographer", "a clockmaker", "a florist",
    ])
    output = _make_input_text(lang, random.randint(1, 2))
    return {
        "lang": lang, "task": "write",
        "instruction": f"Write the opening of a {genre} story set in {setting}. The main character is {character}.",
        "input": "",
        "output": output,
    }


def gen_continue(lang: str) -> dict:
    input_text = _random_fragment(lang)
    output = _random_fragment(lang)
    while output == input_text:
        output = _random_fragment(lang)
    return {
        "lang": lang, "task": "continue",
        "instruction": "Continue this scene in the same style and tone:",
        "input": input_text,
        "output": output,
    }


def gen_humanize(lang: str) -> dict:
    if lang == "it" and HUMANIZE_IT:
        before, after = random.choice(HUMANIZE_IT)
    else:
        before, after = random.choice(HUMANIZE_BEFORE_AFTER)
    return {
        "lang": lang, "task": "humanize",
        "instruction": "Rewrite this to sound more natural and literary:",
        "input": before,
        "output": after,
    }


def gen_title(lang: str) -> dict:
    title, synopsis = random.choice(TITLE_SYNOPSIS_PAIRS)
    if lang == "it":
        inst = f"Proponi un titolo e una tagline per questa sinossi"
    elif lang == "es":
        inst = "Propon un titulo y una tagline para esta sinopsis"
    elif lang == "fr":
        inst = "Propose un titre et une tagline pour ce synopsis"
    elif lang == "de":
        inst = "Schlage einen Titel und eine Tagline fur diese Zusammenfassung vor"
    elif lang == "pt":
        inst = "Proponha um titulo e uma tagline para esta sinopse"
    else:
        inst = "Propose a title and tagline for this synopsis"
    return {
        "lang": lang, "task": "title",
        "instruction": inst,
        "input": synopsis,
        "output": title + (f" — {synopsis.split('.')[0]}" if not title.startswith("Il") else " — Una storia di magia e sacrificio."),
    }


def gen_synopsis(lang: str) -> dict:
    title, synopsis = random.choice(TITLE_SYNOPSIS_PAIRS)
    genre = random.choice(GENRES)
    if lang == "it":
        inst = f"Genera una sinossi avvincente (80-120 parole) per '{title}'"
    elif lang == "es":
        inst = f"Genera una sinopsis atractiva (80-120 palabras) para '{title}'"
    elif lang == "fr":
        inst = f"Redige un synopsis captivant (80-120 mots) pour '{title}'"
    elif lang == "de":
        inst = f"Erstelle eine fesselnde Zusammenfassung (80-120 Wörter) fur '{title}'"
    elif lang == "pt":
        inst = f"Gere uma sinopse atraente (80-120 palavras) para '{title}'"
    else:
        inst = f"Write a compelling synopsis (80-120 words) for '{title}'"
    return {
        "lang": lang, "task": "synopsis",
        "instruction": inst,
        "input": f"Genere: {genre}",
        "output": synopsis,
    }


def gen_seo(lang: str) -> dict:
    title = random.choice(TITLE_SYNOPSIS_PAIRS)[0]
    genre = random.choice(GENRES)
    content = _make_input_text(lang)
    kw = _seo_keywords(content, 5)
    desc = content[:150].replace("\n", " ")
    tags = kw[:3]
    cats = [genre, "Fiction", "Literature"]
    if lang == "it":
        inst = f"Genera metadati SEO in formato JSON per '{title}'"
    elif lang == "es":
        inst = f"Genera metadatos SEO en formato JSON para '{title}'"
    elif lang == "fr":
        inst = f"Genere des metadonnees SEO au format JSON pour '{title}'"
    elif lang == "de":
        inst = f"Generiere SEO-Metadaten im JSON-Format fur '{title}'"
    elif lang == "pt":
        inst = f"Gere metadados SEO em formato JSON para '{title}'"
    else:
        inst = f"Generate SEO metadata in JSON format for '{title}'"
    return {
        "lang": lang, "task": "seo",
        "instruction": inst,
        "input": f"Genere: {genre}. Contenuto: {content[:200]}",
        "output": json.dumps({"keywords": kw, "tags": tags, "description": desc, "categories": cats}, ensure_ascii=False),
    }


def gen_translate(lang: str) -> dict:
    candidates = [p for p in TRANSLATE_PAIRS if p[1] == lang]
    if not candidates:
        candidates = TRANSLATE_PAIRS
    source_text, target_lang, translated = random.choice(candidates)
    if lang == "it":
        inst = f"Traduci in italiano mantenendo il tono letterario"
    elif lang == "es":
        inst = f"Traduce al espanol manteniendo el tono literario"
    elif lang == "fr":
        inst = f"Traduis en francais en maintenant le ton litteraire"
    elif lang == "de":
        inst = f"Ubersetze ins Deutsche unter Beibehaltung des literarischen Tons"
    elif lang == "pt":
        inst = f"Traduza para o portugues mantendo o tom literario"
    else:
        inst = f"Translate this while maintaining the literary tone"
    return {
        "lang": lang, "task": "translate",
        "instruction": inst,
        "input": source_text,
        "output": translated,
    }


GENERATORS = {
    "write": gen_write,
    "continue": gen_continue,
    "humanize": gen_humanize,
    "title": gen_title,
    "synopsis": gen_synopsis,
    "seo": gen_seo,
    "translate": gen_translate,
}


def generate_example(task: str, lang: str, dedup_set: set) -> Optional[dict]:
    gen = GENERATORS.get(task)
    if not gen:
        return None
    ex = gen(lang)
    key = hashlib.md5(json.dumps(ex, ensure_ascii=False).encode()).hexdigest()
    if key in dedup_set:
        return None
    dedup_set.add(key)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-examples", type=int, default=50000, help="Totale esempi da generare")
    ap.add_argument("--out-dir", default="training/data/generated")
    ap.add_argument("--languages", default="it,en,es,fr,de,pt", help="Lingue separate da virgola")
    args = ap.parse_args()

    langs = [l.strip() for l in args.languages.split(",") if l.strip() in LANGUAGES]
    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    dedup = set()
    buffers: Dict[str, List[str]] = {}

    t0 = time.time()
    # Round-robin across tasks and languages
    lang_cycle = itertools.cycle(langs)
    task_cycle = itertools.cycle(TASKS)
    pct = 0

    for i in range(args.num_examples):
        lang = next(lang_cycle)
        task = next(task_cycle)
        ex = generate_example(task, lang, dedup)
        if not ex:
            continue

        key = f"{task}_{lang}"
        line = json.dumps(ex, ensure_ascii=False)
        if key not in buffers:
            buffers[key] = []
        buffers[key].append(line)
        total += 1

        if (i + 1) % max(1, args.num_examples // 20) == 0:
            pct += 5
            elapsed = time.time() - t0
            print(f"  {pct}% ({total} esempi unici, {elapsed:.1f}s)")

    # Write output files
    for key, lines in buffers.items():
        fname = f"generated_{key}.jsonl"
        fpath = os.path.join(args.out_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"  ✓ {fname}: {len(lines)} esempi")

    elapsed = time.time() - t0
    total_files = len(buffers)
    print(f"\nCompletato: {total} esempi in {total_files} file, {elapsed:.1f}s")
    print(f"Output: {args.out_dir}/")


if __name__ == "__main__":
    main()
