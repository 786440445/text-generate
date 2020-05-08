
# 模型保存文件
BASE_MODEL_DIR = 'model'

# 模型名称
MODEL_NAME = 'chatmsg_model.ckpt'

# 有关语料数据的配置
data_config = {
    # 问题最短的长度
    "max_key_len": 10,
    # 问题最长的长度
    "max_chatmsg_len": 20,
    # 词与索引对应的文件
    "word2index_path": "data/w2i.pkl",
    # 原始语料路径
    "corpus_path": "corpus/",
    # 原始语料经过预处理之后的保存路径
    "processed_path": "data/data.pkl",
}

def add_arguments(parser):
    parser.add_argument("--epochs", type=int, default=10, help="Epochs.")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search decoder.")
    parser.add_argument("--embedding_size", type=int, default=100, help="Word embedding size.")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--min_learning_rate", type=float, default=1e-6, help="Min learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden_size.")
    parser.add_argument("--share_embedding", type=bool, default=True, help="Num units.")
    parser.add_argument("--cell_type", type=str, default='gru', help="Num units.")
    parser.add_argument("--layer_size", type=int, default=1, help="Layer size.")
    parser.add_argument("--bidirection", type=bool, default=False, help="Bidirection.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Keep prob.")
    parser.add_argument("--max_decode_step", type=int, default=20, help="Max decode step.")
    parser.add_argument("--dacay_step", type=int, default=10000, help="Decay step.")
    parser.add_argument("--max_gradient_norm", type=float, default=3.0, help="Max gradient norm.")
    parser.add_argument("--coutinue_train", type=bool, default=True, help="Continue Train.")
