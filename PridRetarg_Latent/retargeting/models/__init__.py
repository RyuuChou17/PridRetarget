def create_model(args, character_names, dataset):
    if args.is_pred:
        import models.pred_archi
        print('pred model')
        return models.pred_archi.Pred_model(args, character_names, dataset)
    elif args.model == 'mul_top_mul_ske':
        args.skeleton_info = 'concat'
        import models.architecture
        print('mul_top_mul_ske')
        return models.architecture.GAN_model(args, character_names, dataset)
    else:
        raise Exception('Unimplemented model')
