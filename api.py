from aiohttp import web
import tensorflow as tf


def parse_fn(line):
    # Encode in Bytes for TF
    words = [w.encode() for w in line.strip().split()]

    # Chars
    chars = [[c.encode() for c in w] for w in line.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

    return {'words': [words], 'nwords': [len(words)],
            'chars': [chars], 'nchars': [lengths]}


async def predict(request):
    text = parse_fn(request.rel_url.query['text'])
    prediction = request.app['predict_fn'](text)

    prediction = [tag.decode() for tag in prediction['tags'][0]]
    text = [word.decode() for word in text['words'][0][:len(prediction)]]

    return web.json_response({
        'text': text,
        'tags': prediction})


async def on_startup(app):
    predict_fn = tf.contrib.predictor.from_saved_model('dist_model/')
    app['predict_fn'] = predict_fn


app = web.Application()
app.on_startup.append(on_startup)
app.add_routes([web.get('/predict', predict)])

web.run_app(app)