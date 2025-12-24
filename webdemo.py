import argparse
import torch
from flask import Flask, jsonify, request, render_template
from typing import Optional

from module import init_model_by_key
from module.decoding import DecodeOptions, decode_with_options
class Context(object):
    def __init__(self, path, device: torch.device, max_input_len: int, decode_options: DecodeOptions):
        print(f"loading pretrained model from {path}")
        self.device = device
        self.max_input_len = max_input_len
        self.decode_options = decode_options
        model_info = torch.load(path, map_location=device, weights_only=False)
        self.tokenizer = model_info.get('tokenizer') or model_info.get('tokenzier')
        if self.tokenizer is None:
            raise KeyError("Checkpoint missing tokenizer (expected key 'tokenizer' or legacy 'tokenzier')")
        self.model = init_model_by_key(model_info['args'], self.tokenizer)
        self.model.load_state_dict(model_info['model'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, s):
        if not isinstance(s, str) or not s.strip():
            raise ValueError('missing coupletup')
        if self.max_input_len > 0 and len(s) > self.max_input_len:
            raise ValueError(f'coupletup too long (>{self.max_input_len})')
        input_ids_list = self.tokenizer.encode(s)
        input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids).squeeze(0)
        pred_ids = decode_with_options(logits, input_ids_list, self.tokenizer, self.decode_options)
        pred = self.tokenizer.decode(pred_ids)
        return pred
        
app = Flask(__name__)
ctx: Optional[Context] = None


@app.route('/predict', methods=['POST'])
def predict_api():
    if ctx is None:
        return jsonify({'error': 'model not loaded'}), 500
    payload = request.get_json(silent=True) or {}
    coupletup = payload.get('coupletup')
    try:
        coupletdown = ctx.predict(coupletup)
        return jsonify({'coupletdown': coupletdown})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    if ctx is None:
        return render_template("index.html", error="model not loaded")
    coupletup = request.form.get("coupletup")
    try:
        coupletdown = ctx.predict(coupletup)
        return render_template("index.html", coupletdown=coupletdown)
    except ValueError as e:
        return render_template("index.html", error=str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--max_input_len', default=64, type=int)
    parser.add_argument('--decode', default='constrained', choices=['argmax', 'constrained', 'beam'], type=str)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--no_copy', action='store_true', default=False)
    parser.add_argument('--max_repeat', default=2, type=int)
    parser.add_argument('--match_punct', action='store_true', default=False)
    parser.add_argument('--host', default='0.0.0.0', type=str)
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    decode_opts = DecodeOptions(
        strategy=args.decode,
        topk=args.topk,
        beam_size=args.beam_size,
        no_copy=bool(args.no_copy),
        max_repeat=int(args.max_repeat),
        match_punct=bool(args.match_punct),
    )
    ctx = Context(args.model, device=device, max_input_len=args.max_input_len, decode_options=decode_opts)
    app.run(host=args.host, port=args.port)
