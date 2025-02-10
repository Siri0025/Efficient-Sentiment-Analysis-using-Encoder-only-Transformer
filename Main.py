from Transformer_build import *
from Train_build import *
import pandas as pd
import optuna
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(features, labels, num_features,num_classes):
    def objective(trial):
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        lr = trial.suggest_float('lr', 0.0001, 0.01)
        batch_size = trial.suggest_categorical('batch_size', [16,24,32,64])
        n_heads = trial.suggest_categorical('num_heads', [1,2,4,8])
        d_model = trial.suggest_categorical('d_model', [256,512])
        d_ff = trial.suggest_categorical('d_ff', [1024,2048])
        N_module = trial.suggest_int('N_module',1,6,step=1)
        n_epochs = trial.suggest_categorical('n_epochs',[100,200,300,400])
        if d_model % n_heads != 0:
            raise optuna.exceptions.TrialPruned()
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, d_model, dropout) 
        ff = PositionwiseFeedForward(d_model, d_ff, dropout) 
        encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        encoder = Encoder(encoder_layer, N_module, d_model , dropout, num_features,num_classes=num_classes).cuda()  
        if opt.data_parallel: 
            encoder = torch.nn.DataParallel(encoder).cuda()
        best_accuracy,test_loss,test_precision,test_recall,test_f1,val_loss,val_precision,val_recall,val_f1,test_accuracy,val_accuracy = train(features, labels, encoder,lr,batch_size,n_epochs)

        return best_accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best parameters:", study.best_params)
    print("Best Overall Accuracy:", study.best_value)

    best_params = study.best_params
    dropout = best_params['dropout']
    lr = best_params['lr']
    batch_size = best_params['batch_size']
    n_heads = best_params['num_heads']
    d_model = best_params['d_model']
    d_ff = best_params['d_ff']
    N_module = best_params['N_module']
    n_epochs = best_params['n_epochs']

    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, d_model, dropout) 
    ff = PositionwiseFeedForward(d_model, d_ff, dropout) 
    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    encoder = Encoder(encoder_layer, N_module, d_model , dropout, num_features,num_classes=num_classes).cuda() 
    if opt.data_parallel: 
        encoder = torch.nn.DataParallel(encoder).cuda()
    best_accuracy,test_loss,test_precision,test_recall,test_f1,val_loss,val_precision,val_recall,val_f1,test_accuracy,val_accuracy = train(features, labels, encoder,lr,batch_size,n_epochs)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Validate Loss: {val_loss:.4f}')
    print(f'Accuracy: {test_accuracy:.4f}')
    print(f'Val accuracy: {val_accuracy:.4f}')
    print(f'Test_Precision: {test_precision:.4f}')
    print(f'Val_Precision: {val_precision:.4f}')
    print(f'test_Recall: {test_recall:.4f}')
    print(f'Val_Recall: {val_recall:.4f}')
    print(f'Test_F1 Score: {test_f1:.4f}')
    print(f'Val_F1 Score: {val_f1:.4f}')
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    print("Best parameters:", study.best_params)
    print("Best Overall accuracy:", study.best_value)


if __name__ == '__main__':
    train_features = pd.read_csv(opt.data_dir + '/train_features.csv', header=0, index_col=False).to_numpy()
    test_features = pd.read_csv(opt.data_dir + '/test_features.csv', header=0, index_col=False).to_numpy()
    val_features = pd.read_csv(opt.data_dir + '/validate_features.csv', header=0, index_col=False).to_numpy()
    features = [train_features, test_features,val_features]

    train_labels = pd.read_csv(opt.data_dir + '/train.csv', header=0, index_col=False).to_numpy()
    test_labels = pd.read_csv(opt.data_dir + '/test.csv', header=0, index_col=False).to_numpy()
    val_labels = pd.read_csv(opt.data_dir + '/validate.csv', header=0, index_col=False).to_numpy()
    labels = [train_labels, test_labels,val_labels]

    num_features = train_features.shape[1]-1
    print('train features shape', train_features.shape)
    print('train labels shape', train_labels.shape)
    print('val features shape', val_features.shape)
    print('val labels shape', val_labels.shape)
    print('test features shape', test_features.shape)
    print('test labels shape', test_labels.shape)
    


    print('num features', num_features)
    total_data = train_features.shape[0]  + test_features.shape[0] + val_features.shape[0]
    print('total data', total_data)
    print()

    print('train max label', train_labels[:].max())
    print('train min label', train_labels[:].min())
    print('val max label', val_labels[:].max())
    print('val min label', val_labels[:].min())
    print('test max label', test_labels[:].max())
    print('test min label', test_labels[:].min())

    print("Features shape:", len(features))
    print("Labels shape:", len(labels))

    print(f"Features type: {features[0].dtype}")
    print(f"Labels type: {labels[0].dtype}")

    features = [f.astype(np.float32) for f in features]
    labels = [l.astype(np.float32) for l in labels]
    num_classes = 3
    print(f"Features shape: {[f.shape for f in features]}")
    print(f"Labels shape: {[l.shape for l in labels]}")

    main(features, labels, num_features,num_classes)