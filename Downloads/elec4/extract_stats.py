import csv, json, re, random
from collections import Counter

rows = []
with open('dataset.csv', encoding='utf-8', errors='ignore') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Sentiment distribution
sent_dist = dict(Counter(r['Sentiment'] for r in rows if r.get('Sentiment')))
print('Sentiment:', sent_dist)

# Category distribution
cat_dist = dict(Counter(r['Category'] for r in rows if r.get('Category')))
print('Categories:', cat_dist)

# Top words per emotion
STOPWORDS = {'the','a','an','is','it','was','in','of','to','and','for','this',
             'that','are','be','with','i','my','on','at','from','so','not','but',
             'have','had','has','we','they','very','its','or','as','if','by','do',
             'no','up','all','can','get','just','also','more','been','their','when',
             'one','out','too','did','got','will','about','into','already','after',
             'still','would','could','they', 'our', 'your', 'you', 'am', 'me', 'him',
             'her', 'any', 'even', 'only'}

def top_words(subset, n=20):
    counts = Counter()
    for r in subset:
        text = r.get('Customer Review (English)', '')
        words = re.findall(r"[a-z']+", text.lower())
        words = [w.strip("'") for w in words if w.strip("'") not in STOPWORDS and len(w.strip("'")) > 2]
        counts.update(words)
    return counts.most_common(n)

all_rows = rows
happy_rows = [r for r in rows if r['Emotion'] == 'Happy']
sad_rows   = [r for r in rows if r['Emotion'] == 'Sadness']
anger_rows = [r for r in rows if r['Emotion'] == 'Anger']
love_rows  = [r for r in rows if r['Emotion'] == 'Love']
fear_rows  = [r for r in rows if r['Emotion'] == 'Fear']

all_top   = top_words(all_rows)
happy_top = top_words(happy_rows)
sad_top   = top_words(sad_rows)
anger_top = top_words(anger_rows)
love_top  = top_words(love_rows)
fear_top  = top_words(fear_rows)

print('All top words:', all_top)
print('Happy:', happy_top)
print('Sad:', sad_top)
print('Anger:', anger_top)
print('Love:', love_top)
print('Fear:', fear_top)

# Sample reviews
random.seed(42)
samples = []
for emo in ['Happy', 'Sadness', 'Anger', 'Love', 'Fear']:
    pool = [r for r in rows if r['Emotion'] == emo]
    picked = random.sample(pool, min(2, len(pool)))
    for r in picked:
        samples.append({
            'Category': r['Category'],
            'Emotion': r['Emotion'],
            'CustomerRating': r['Customer Rating'],
            'Review': r['Customer Review (English)'][:160]
        })

print('Samples:', json.dumps(samples, indent=2))
