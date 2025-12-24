import torch
from module import Tokenizer, init_model_by_key
import argparse
from module.decoding import DecodeOptions, decode_with_options

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, type=str)
    parser.add_argument("-s", "--stop_flag", default='q', type=str)
    parser.add_argument("-c", "--cuda", action='store_true')
    parser.add_argument("--decode", default='constrained', choices=['argmax', 'constrained', 'beam'], type=str)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--no_copy", action='store_true', default=False)
    parser.add_argument("--max_repeat", default=2, type=int)
    parser.add_argument("--match_punct", action='store_true', default=False)
    args = parser.parse_args()
    print("loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model_info = torch.load(args.path)
    tokenizer = model_info.get('tokenizer') or model_info.get('tokenzier')
    if tokenizer is None:
        raise KeyError("Checkpoint missing tokenizer (expected key 'tokenizer' or legacy 'tokenzier')")
    model = init_model_by_key(model_info['args'], tokenizer)
    model.load_state_dict(model_info['model'])
    model.to(device)
    model.eval()

    decode_opts = DecodeOptions(
        strategy=args.decode,
        topk=args.topk,
        beam_size=args.beam_size,
        no_copy=bool(args.no_copy),
        max_repeat=int(args.max_repeat),
        match_punct=bool(args.match_punct),
    )
    while True:
        question = input("上联：")
        if question == args.stop_flag.lower():
            print("Thank you!")
            break
        input_ids_list = tokenizer.encode(question)
        input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids).squeeze(0)
        pred_ids = decode_with_options(logits, input_ids_list, tokenizer, decode_opts)
        pred = tokenizer.decode(pred_ids)
        print(f"下联：{pred}")
if __name__ == "__main__":
    run()
