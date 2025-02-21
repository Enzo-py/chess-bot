from abc import ABC, abstractmethod
from src.utils.console import Style

__all__ = ["TrainConfig", "TrainConfigBase", "AllHeads", "GenerativeHead", "BoardEvaluationHead", "EncoderHead", "OnPuzzles", "OnGames", "WithPrints", "AutoSave"]

class TrainConfig:
    """Configuration de l'entraînement via `|`."""

    line_length = 50

    class UndefinedConfigError(Exception):
        def __init__(self, configs, *args):
            super().__init__(*args)
            self.message = "[TrainConfig] Configuration not defined \n|__ You must define a configuration for <" + ", ".join(configs) + ">"

        def __str__(self):
            return Style("ERROR", self.__repr__() + "\n" + self.message)

    def __init__(self, engine):
        self.engine = engine

    def __enter__(self) -> 'TrainConfig':
        """Démarre un contexte de training."""
        
        if self.engine._train_config["UI"] == 'prints':
            print('╭' + '─'*self.line_length + '╮')
            line_left = "🔄 TRAINING SESSION "
            line_right = f"Engine: <{self.engine.__class__.__name__}>"

            print(f"| {(line_left + line_right):<{self.line_length-3}} |")
            line = "Mode: <" + self.engine._train_config["mode"] + "> | Head: <" + self.engine._train_config["head"] + ">"
            print(f"| {line: ^{self.line_length-2}} |")
            print('|' + '─'*self.line_length + '|')

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Fin du contexte."""

        # check if error
        if exc_type is not None:
            print("❌" + Style("ERROR", f"An error occured: {exc_value}"))
            print("_"*50)
        
        if self.engine._train_config["UI"] == 'prints':
            line = "✅ Training session completed"
            print("|" + "─"*self.line_length + "|")
            print(f"| {line: <{self.line_length-3}} |")
            print("╰" + "─"*self.line_length + "╯")

    def __or__(self, value: 'TrainConfigBase') -> 'TrainConfig':
        """Applique une configuration."""
        return value.apply(self.engine)

    def train(self, **data):
        """
            Exécute l'entraînement. ---------------
            wow
        """
        undefined_config = []
        if self.engine._train_config["head"] is None:
            undefined_config.append("head")
        if self.engine._train_config["mode"] is None:
            undefined_config.append("mode")
        
        if len(undefined_config) > 0:
            raise TrainConfig.UndefinedConfigError(undefined_config)
        
        if self.engine._train_config["UI"] == 'prints':
            line = ">> Start training"
            print(f"| {line: <{self.line_length-2}} |")

        if self.engine._train_config["mode"] == "puzzles":
            output = self.engine._train_on_puzzles(**data)
        elif self.engine._train_config["mode"] == "games":
            output = self.engine._train_on_games(**data)
        else:
            raise ValueError("Mode not unknown")
        
        return output
        

    def test(self, _plot=False):
        """Évalue le modèle."""
        self.engine.test(_plot=_plot)

    @property
    def print(self):
        """Affiche la configuration."""
        self._print = True
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

class OnPuzzles(TrainConfigBase):
    """Configuration pour entraîner sur les puzzles."""

    def apply(self, engine) -> TrainConfig:
        if engine._train_config["mode"] is not None:
            raise TrainConfigBase.OverwriteError(f"<mode={engine._train_config['mode']}>")
        
        engine._train_config["mode"] = "puzzles"
        return TrainConfig(engine)

class OnGames(TrainConfigBase):
    """Configuration pour entraîner sur les parties complètes."""

    def apply(self, engine) -> TrainConfig:
        if engine._train_config["mode"] is not None:
            raise TrainConfigBase.OverwriteError(f"<mode={engine._train_config['mode']}>")
        
        engine._train_config["mode"] = "games"
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