from encoder import Model
model = Model()

def handle(sentence):
    # text = [
    #     'the food was hot and the service was fast!',
    #     'worst thing ever, i hate everything about this, will return immediately',
    #     'dead on arrival',
    #     'this is a life saver, i cant live without this',
    #     'it was okay, nothing amazing, but nothing to complain about either'
    # ]
    text_features = model.transform([sentence])
    return text_features[0, 2388]
