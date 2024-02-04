import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

# from data.scientific_papers import ScientificPapersDataModule
# from models.SimpleTransformer import SimpleTransformer, SimpleTransformerConfig
from utils.parser import parser

# def train(args):
#     dm = ScientificPapersDataModule(
#         dataset_variant=args.dataset_variant,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         tokenizer_name=args.model_name,
#         dataset_limit=args.dataset_limit,
#     )
#
#     SimpleTransformerConfig.register_for_auto_class()
#     SimpleTransformer.register_for_auto_class("AutoModelForSeq2SeqLM")
#
#     config = SimpleTransformerConfig()
#     model = SimpleTransformer(config)


if __name__ == "__main__":
    args = parser()
    # train(args)
