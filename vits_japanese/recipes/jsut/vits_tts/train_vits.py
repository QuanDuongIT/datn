import os
import argparse
from glob import glob

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=True, help="Number of training epochs.")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV file.")
    parser.add_argument("--output_path", required=True, help="Path to output directory.")
    parser.add_argument("--restore_path", required=False, default=None, help="Path to pretrained model .pth file (optional).")
    parser.add_argument("--continue_path", required=False, default=None, help="Path to continue training from previous folder.")
    args = parser.parse_args()

    dataset_config = BaseDatasetConfig(
        formatter="jsut",
        meta_file_train=args.metadata,
        path=os.path.dirname(args.metadata)
    )

    audio_config = VitsAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    vitsArgs = VitsArgs(
        use_language_embedding=False,
        use_speaker_embedding=False,
        use_sdp=False,
    )

    config = VitsConfig(
        model_args=vitsArgs,
        audio=audio_config,
        run_name="vits_jsut",
        use_speaker_embedding=False,
        save_step=1000, 
        eval_step=100, 
        save_n_checkpoints=1,
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=0,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner="ja_jp_phonemizer",
        use_phonemes=True,
        phoneme_language="ja-jp",
        phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        use_language_weighted_sampler=False,
        print_eval=False,
        mixed_precision=False,
        min_audio_len=32 * 256 * 4,
        max_audio_len=160000,
        output_path=args.output_path,
        datasets=[dataset_config],
        characters=CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad="_",
            eos="",
            bos="",
            blank="",
            characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            punctuations=";:,.!?¡¿—…\"«»“” ",
            phonemes="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ",
        ),
        test_sentences=[
            [
                "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                None,
                None,
                "ja-JP",
            ]
        ],
    )

    ap = AudioProcessor.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    config.model_args.num_speakers = 1
    config.model_args.num_languages = 1

    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer)

    # Chọn logic giữa continue và restore
    if args.continue_path and os.path.isdir(args.continue_path):
        trainer_args = TrainerArgs(continue_path=args.continue_path)
        print(f"[INFO] Continuing training from folder: {args.continue_path}")
    elif args.restore_path and os.path.exists(args.restore_path):
        trainer_args = TrainerArgs(restore_path=args.restore_path)
        print(f"[INFO] Restoring model from checkpoint: {args.restore_path}")
    else:
        trainer_args = TrainerArgs()
        print("[INFO] No checkpoint provided. Training from scratch.")

    trainer = Trainer(
        trainer_args, config, args.output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    trainer.fit()

if __name__ == "__main__":
    main()
