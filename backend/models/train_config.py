from abc import ABC, abstractmethod
from src.utils.console import Style
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["TrainConfig", "TrainConfigBase", "AllHeads", "GenerativeHead", "BoardEvaluationHead", "EncoderHead", "WithPrints", "AutoSave"]

class TrainConfig:
    """Train configuration manager."""

    line_length = 80

    class UndefinedConfigError(Exception):
        def __init__(self, configs, *args):
            super().__init__(*args)
            self.message = "[TrainConfig] Configuration not defined \n|__ You must define a configuration for <" + ", ".join(configs) + ">"

        def __str__(self):
            return Style("ERROR", self.__repr__() + "\n" + self.message)

    def __init__(self, engine):
        self.engine = engine

    def __enter__(self) -> 'TrainConfig':
        """Start the training session."""
        
        if self.engine._train_config["UI"] == 'prints':
            print('╭' + '─'*self.line_length + '╮')
            line_left = "🔄 TRAINING SESSION "
            line_right = f"Engine: <{self.engine.__class__.__name__}>"

            print(f"| {(line_left + line_right):<{self.line_length-3}} |")
            line = f"On <{self.engine._train_config['mode']}> with <{self.engine._train_config['head']}> head"
            print(f"| {line: <{self.line_length-2}} |")
            print('|' + '─'*self.line_length + '|')

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """End the training session."""

        # check if error
        if exc_type is not None:
            msg = Style(None, f"An error occured: {exc_value.__str__()}", auto_break=True, max_length=self.line_length-2).__str__()
            lines = msg.split("\n")
            print("|" + "─"*self.line_length + "|")
            for line in lines:
                print(f"| {Style('ERROR', line.strip()).__str__(): <{self.line_length+7}} |")
            print("╰" + "─"*self.line_length + "╯")
            return
        
        if self.engine._train_config["UI"] == 'prints':
            line = "✅ Training session completed"
            print("|" + "─"*self.line_length + "|")
            print(f"| {line: <{self.line_length-3}} |")
            print("╰" + "─"*self.line_length + "╯")

        # remove config from engine
        self.engine._train_config["head"] = None
        self.engine._train_config["mode"] = None
        self.engine._train_config["UI"] = None
        self.engine._train_config["auto_save"] = False
        self.engine._train_config["load_policy"] = "all"

    def __or__(self, value: 'TrainConfigBase') -> 'TrainConfig':
        """Applique une configuration."""
        return value.apply(self.engine)

    def train(self, epochs=10, batch_size=16, loader=None):
        """
            Train the model.
            You need to provide the games and the labels OR a loader.
        """
        undefined_config = []
        if self.engine._train_config["head"] is None:
            undefined_config.append("head")
        
        if len(undefined_config) > 0:
            raise TrainConfig.UndefinedConfigError(undefined_config)
        
        if self.engine._train_config["UI"] == 'prints':
            line = ">> Start training for " + str(epochs) + " epochs"
            print(f"| {line: <{self.line_length-2}} |")

        if loader is None: raise Exception("No data loader ?")

        self.engine._get_validation_set(loader, batch_size * 6)
        if self.engine._train_config["head"] == "all":
            raise NotImplementedError("Training all heads is not implemented yet.")
        elif self.engine._train_config["head"] == "generative":
            return self.engine._train_on_generation(epochs, batch_size, loader=loader)
        
        elif self.engine._train_config["head"] == "board_evaluation":
            return self.engine._train_board_evaluation(epochs, batch_size, loader=loader)
        
        elif self.engine._train_config["head"] == "encoder":
            return self.engine._train_encoder(epochs, batch_size, loader=loader)
        
        else:
            raise ValueError(f"Invalid head: {self.engine._train_config['head']}")
        

    def test(self, _plot=False, **data):
        """Evaluate the model."""
        
        undefined_config = []
        if self.engine._train_config["head"] is None:
            undefined_config.append("head")

        if len(undefined_config) > 0:
            raise TrainConfig.UndefinedConfigError(undefined_config)
        
        if self.engine._train_config["UI"] == 'prints':
            line = ">> Start testing"
            print(f"| {line: <{self.line_length-2}} |")

        games = data.get("games", None)
        best_moves = data.get("best_moves", None) or data.get("moves", None)
        loader = data.get("loader", None)

        if loader is None:
            assert games is not None, "You need to provide games to test on."
            assert best_moves is not None or self.engine._train_config["head"] == "encoder", "You need to provide best moves to test on."

        self.engine.module.eval()
        if self.engine._train_config["head"] == "all":
            raise NotImplementedError("Testing all heads is not implemented yet.")
        elif self.engine._train_config["head"] == "generative":
            return self.engine._test_generation(games, best_moves, loader=loader)
        elif self.engine._train_config["head"] == "board_evaluation":
            return self.engine._test_board_evaluation(games, best_moves, loader=loader)
        elif self.engine._train_config["head"] == "encoder":
            return self.engine._test_encoder(games, loader=loader)
        else:
            raise ValueError(f"Invalid head: {self.engine._train_config['head']}")
        
    def plot(self, data, title=None):
        if len(data) < 5:
            raise Exception("Invalid data, should come from env.train")
        train_loss, val_loss, train_acc, val_acc = data[0], data[1], data[2], data[3]
        if title is None:
            title = data[-1]

        epochs = range(1, len(train_loss) + 1)
        
        # Set a beautiful seaborn style
        sns.set_theme(style="whitegrid")
        
        # Create subplots: one for loss and one for accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot loss curves on the first axis
        sns.lineplot(x=epochs, y=train_loss, ax=ax1, label="Train Loss",
                    color="blue", marker="o")
        sns.lineplot(x=epochs, y=val_loss, ax=ax1, label="Val Loss",
                    color="orange", marker="o")
        ax1.set_xlabel("Epochs", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Loss over Epochs", fontsize=14)
        
        # Plot accuracy curves on the second axis
        sns.lineplot(x=epochs, y=train_acc, ax=ax2, label="Train Acc",
                    color="blue", marker="o")
        sns.lineplot(x=epochs, y=val_acc, ax=ax2, label="Val Acc",
                    color="orange", marker="o")
        ax2.set_xlabel("Epochs", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Accuracy over Epochs", fontsize=14)
        
        # Set an overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust legends
        ax1.legend(loc="upper right", fontsize=10)
        ax2.legend(loc="upper right", fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()



    def set_load_policy(self, policy):
        """
            Define the loading policy of the games.
            The policy can be:
                - "all": load all type of games
                - "early-game": load only games at the early stage (first 10 moves)
                - "mid-game": load only games at the mid stage (10 to 30 moves)
                - "end-game": load only games at the end stage (more than 30 moves)
        """
        self.engine._train_config["load_policy"] = policy
        return self

class TrainConfigBase(ABC):
    """Base pour les configurations de training."""

    class OverwriteError(Exception):
        def __init__(self, config, *args):
            super().__init__(*args)
            self.message = f"[TrainConfig] Configuration already set: <{config}>"
        
        def __str__(self):
            return Style("ERROR", self.__repr__() + "\n" + self.message + "\n You can't overwrite a configuration.")

    @abstractmethod
    def apply(self, engine) -> TrainConfig:
        raise NotImplementedError


class AllHeads(TrainConfigBase):
    """Configuration to train all heads."""

    def apply(self, engine) -> TrainConfig:
        if engine._train_config["head"] is not None:
            raise TrainConfigBase.OverwriteError(f"<head={engine._train_config['head']}>")
        
        engine._train_config["head"] = "all"
        return TrainConfig(engine)
    
class GenerativeHead(TrainConfigBase):
    """Configuration to train the generative head."""
    
    def apply(self, engine) -> TrainConfig:
        if engine._train_config["head"] is not None:
            raise TrainConfigBase.OverwriteError(f"<head={engine._train_config['head']}>")
        
        engine._train_config["head"] = "generative"
        return TrainConfig(engine)
    
class BoardEvaluationHead(TrainConfigBase):
    """Configuration to train the board evaluation head."""
    
    def apply(self, engine) -> TrainConfig:
        if engine._train_config["head"] is not None:
            raise TrainConfigBase.OverwriteError(f"<head={engine._train_config['head']}>")
        
        engine._train_config["head"] = "board_evaluation"
        return TrainConfig(engine)

class EncoderHead(TrainConfigBase):
    """Configuration to train the encoder head."""
    
    def apply(self, engine) -> TrainConfig:
        if engine._train_config["head"] is not None:
            raise TrainConfigBase.OverwriteError(f"<head={engine._train_config['head']}>")
        
        engine._train_config["head"] = "encoder"
        return TrainConfig(engine)

class WithPrints(TrainConfigBase):
    """Configuration to print the training steps."""
    
    def apply(self, engine) -> TrainConfig:
        if engine._train_config["UI"] is not None:
            raise TrainConfigBase.OverwriteError(f"<UI={engine._train_config['prints']}>")
        
        engine._train_config["UI"] = 'prints'
        return TrainConfig(engine)
    
class AutoSave(TrainConfigBase):

    def apply(self, engine) -> TrainConfig:
        engine._train_config["auto_save"] = True
        return TrainConfig(engine)
