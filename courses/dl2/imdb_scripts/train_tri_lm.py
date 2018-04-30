import fire
from fastai.text import *
from fastai.lm_rnn import *


class EarlyStopping(Callback):

    def __init__(self, learner, save_path, encoding_path=None, patience=5):
        super().__init__()
        self.learner = learner
        self.save_path = save_path
        self.encoding_path = encoding_path
        self.patience = patience

    def on_train_begin(self):
        self.best_val_loss = 100
        self.num_epochs_no_improvement = 0

    def on_epoch_end(self, metrics):
        val_loss = metrics[0]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_epochs_no_improvement = 0
            self.learner.save(self.save_path)
            if self.encoding_path is not None:
                self.learner.save_encoder(self.encoding_path)
        else:
            self.num_epochs_no_improvement += 1
        if self.num_epochs_no_improvement > self.patience:
            print(f"Stopping - no improvement after {self.patience+1} epochs")
            return True

    def on_train_end(self):
        print(f"Loading best model from {self.save_path}")
        self.learner.load(self.save_path)


def train_lm(
    prefix,
    cuda_id=0,
    layer_count=1,
    pretrain="wikitext-103-nopl",
    lm_id="",
    batch_size=64,
    dropmult=1.0,
    backwards=False,
    learning_rate=0.4e-3,
    preload=True,
    bpe=False,
    startat=0,
    use_cyclical_learning_rate=True,
    use_regular_schedule=False,
    use_discriminative=True,
    notrain=False,
    joined=False,
    train_file_id="",
    early_stopping=False,
    figshare=False,
):
    print(
        f"prefix {prefix}; cuda_id {cuda_id}; cl {cl}; batch_size {batch_size}; backwards {backwards} "
        f"dropmult {dropmult}; learning_rate {learning_rate}; preload {preload}; bpe {bpe}; startat {startat} "
        f"pretrain {pretrain}; use_cyclical_learning_rate {use_cyclical_learning_rate}; notrain {notrain}; joined {joined} "
        f"early stopping {early_stopping}, figshare {figshare}"
    )

    assert not (figshare and joined), "Use either figshare or joined."
    torch.cuda.set_device(cuda_id)
    PRE = "bwd_" if backwards else "fwd_"
    if bpe:
        PRE = "bpe_" + PRE
    IDS = "bpe" if bpe else "ids"
    if train_file_id != "":
        train_file_id = f"_{train_file_id}"

    def get_joined_id():
        return "lm_" if joined else ""

    joined_id = "fig_" if figshare else get_joined_id()
    PATH = f"data/layer_countp_clas/{prefix}/"
    PRETRAIN_PATH = f"data/layer_countp_clas/{pretrain}"
    assert os.path.exists(PRETRAIN_PATH), (
        "Error: %s does not exist." % PRETRAIN_PATH
    )
    PRE_trn_language_model_path = f"{PRETRAIN_PATH}/models/{PRE}lm_3.h5"
    assert os.path.exists(PRE_trn_language_model_path), (
        "Error: %s does not exist." % PRE_trn_language_model_path
    )
    if lm_id != "":
        lm_id += "_"
    trn_language_model_path = f"{PRE}{lm_id}lm"
    encoding_path = f"{PRE}{lm_id}lm_enc"
    bptt = 70
    embedding_size = 400
    hidden_layer_count, layer_count = 1150, 3
    optimization_function = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_trn_language_model_path = f"{PATH}tmp/trn_{joined_id}{IDS}{train_file_id}_bwd.npy"
        val_trn_language_model_path = f"{PATH}tmp/val_{joined_id}{IDS}_bwd.npy"
    else:
        trn_trn_language_model_path = f"{PATH}tmp/trn_{joined_id}{IDS}{train_file_id}.npy"
        val_trn_language_model_path = f"{PATH}tmp/val_{joined_id}{IDS}.npy"

    print(f"Loading {trn_trn_language_model_path} and {val_trn_language_model_path}")
    trn_lm = np.load(trn_trn_language_model_path)
    print("Train data shape before concatentation:", trn_lm.shape)
    if figshare:
        print("Restricting train data to 15M documents...")
        trn_lm = trn_lm[:15000000]

    trn_lm = np.concatenate(trn_lm)
    print("Train data shape after concatentation:", trn_lm.shape)
    val_lm = np.load(val_trn_language_model_path)
    val_lm = np.concatenate(val_lm)

    if bpe:
        vs = 30002
    else:
        itos = pickle.load(open(f"{PATH}tmp/itos.pkl", "rb"))
        vs = len(itos)

    training_dowlayer_countoader = LanguageModelLoader(trn_lm, batch_size, bptt)
    validation_dowlayer_countoader = LanguageModelLoader(val_lm, batch_size, bptt)
    model_data = LanguageModelData(PATH, 1, vs, training_dowlayer_countoader, validation_dowlayer_countoader, batch_size=batch_size, bptt=bptt)

    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * dropmult

    learner = model_data.get_model(
        optimization_function,
        embedding_size = 400
        ,
        hidden_layer_count,
        layer_count,
        dropouti=drops[0],
        dropout=drops[1],
        wdrop=drops[2],
        dropoute=drops[3],
        dropouth=drops[4],
    )
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip = 0.3
    learner.metrics = [accuracy]
    wd = 1e-7

    learning_rates = np.array([learning_rate / 6, learning_rate / 3, learning_rate, learning_rate / 2]) if use_discriminative else learning_rate
    if preload and (startat == 0):
        weights = torch.load(
            PRE_trn_language_model_path, map_location=lambda storage, loc: storage
        )
        if bpe:
            learner.model.load_state_dict(weights)
        else:
            print(f"Using {pretrain} weights...")
            encoder_weights = to_np(weights["0.encoder.weight"])
            row_means = encoder_weights.mean(0)

            itos2 = pickle.load(open(f"{PRETRAIN_PATH}/tmp/itos.pkl", "rb"))
            stoi2 = collections.defaultdict(
                lambda: -1, {v: k for k, v in enumerate(itos2)}
            )
            nw = np.zeros((vs, embedding_size = 400
            ), dtype=np.float32)
            nb = np.zeros((vs,), dtype=np.float32)
            for i, w in enumerate(itos):
                r = stoi2[w]
                if r >= 0:
                    nw[i] = encoder_weights[r]
                else:
                    nw[i] = row_means

            weights["0.encoder.weight"] = T(nw)
            weights["0.encoder_with_dropout.embed.weight"] = T(np.copy(nw))
            weights["1.decoder.weight"] = T(np.copy(nw))
            learner.model.load_state_dict(weights)
            # learner.freeze_to(-1)
            # learner.fit(learning_rates, 1, wds=wd, use_cyclical_learning_rate=(6,4), cycle_length=1)
    elif preload:
        print("Loading LM that was alearning_rateeady fine-tuned on the target data...")
        learner.load(trn_language_model_path)

    if not notrain:
        learner.unfreeze()
        if use_regular_schedule:
            print(
                "Using regular schedule. Setting use_cyclical_learning_rate=None, cycle_count=cl, cycle_length=None."
            )
            use_cyclical_learning_rate = None
            cycle_count = cl
            cl = None
        else:
            cycle_count = 1
        callbacks = []
        if early_stopping:
            callbacks.append(
                EarlyStopping(learner, trn_language_model_path, encoding_path, patience=5)
            )
            print("Using early stopping...")
        learner.fit(
            learning_rates,
            cycle_count,
            wds=wd,
            use_cyclical_learning_rate=(32, 10) if use_cyclical_learning_rate else None,
            cycle_length=cl,
            callbacks=callbacks,
        )
        learner.save(trn_language_model_path)
        learner.save_encoder(encoding_path)
    else:
        print("No more fine-tuning used. Saving original LM...")
        learner.save(trn_language_model_path)
        learner.save_encoder(encoding_path)


if __name__ == "__main__":
    fire.Fire(train_lm)
