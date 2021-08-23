def set_template(args):
    if args.noise == 'real':
        args.lr = 0.0001 
        args.lambda_ratio = 1

    if args.noise == 'synth':
        args.lr = 0.0003 
        args.lambda_ratio = 2
