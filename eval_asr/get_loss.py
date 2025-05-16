import os
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class LogAnalyzer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.ckpt_files = os.listdir(log_dir)
        self.existing_ckpt_steps = self._get_existing_ckpt_steps()
        self.event_acc = EventAccumulator(log_dir)
        self.event_acc.Reload()
        self.available_tags = self.event_acc.Tags().get('scalars', [])
        self.important_tags = self._filter_tags()
        self.logs_dict = self._build_logs_dict()
        self.all_logged_steps = sorted(self.logs_dict.keys())

    def _get_existing_ckpt_steps(self):
        steps = []
        for f in self.ckpt_files:
            match = re.match(r'G_(\d+)\.pth', f)
            if match:
                step = int(match.group(1))
                if step % 1000 == 0:
                    steps.append(step)
        return set(steps)

    def _filter_tags(self):
        tags = [
            'loss/g/total',
            'loss/d/total',
            'loss/g/mel',
            'loss/g/fm',
            'loss/g/kl',
            'loss/g/dur',
            'learning_rate',
            'grad_norm_g',
            'grad_norm_d'
        ]
        return [tag for tag in tags if tag in self.available_tags]

    def _build_logs_dict(self):
        logs = {}
        for tag in self.important_tags:
            for event in self.event_acc.Scalars(tag):
                if event.step % 1000 == 0:
                    if event.step not in logs:
                        logs[event.step] = {'step': event.step}
                    logs[event.step][tag] = event.value
        return logs

    def search_logs_by_checkpoint(self, step):
        if step in self.logs_dict:
            print(f"\n================== Step {step} ==================")
            print(f"Checkpoint: {'(tồn tại)' if step in self.existing_ckpt_steps else '(đã xóa)'}")
            for tag in self.important_tags:
                print(f"{tag}: {self.logs_dict[step].get(tag, '(no log at this step)')}")
            
            # Trả về dict các giá trị loss cần thiết
            return {
                "loss_g_total": self.logs_dict[step].get("loss/g/total", None),
                "loss_d_total": self.logs_dict[step].get("loss/d/total", None),
                "loss_mel": self.logs_dict[step].get("loss/g/mel", None),
                "loss_fm": self.logs_dict[step].get("loss/g/fm", None),
                "loss_g_kl": self.logs_dict[step].get("loss/g/kl", None),
            }
        else:
            print(f"Không tìm thấy thông tin log cho step {step}")
            return {"loss_total": None, "loss_mel": None, "loss_fm": None}

    def print_logs_from_dict(self):
        for step in self.all_logged_steps:
            print(f"\n================== Step {step} ==================")
            print(f"Checkpoint: {'(tồn tại)' if step in self.existing_ckpt_steps else '(đã xóa)'}")
            for tag in self.important_tags:
                print(f"{tag}: {self.logs_dict[step].get(tag, '(no log at this step)')}")

