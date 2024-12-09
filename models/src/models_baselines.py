import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score

# Define GPU XGBoost Classifier Model
class GPUXGBoostModelBC(nn.Module):
    def __init__(self, max_depth=7, learning_rate=0.1, n_estimators=100):
        super(GPUXGBoostModelBC, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.gb_model = xgb.XGBClassifier(
            device=self.device,
            max_depth=self.max_depth, 
            learning_rate=self.learning_rate, # default is 0.3
            n_estimators=self.n_estimators,
            tree_method='auto',
            objective='binary:hinge',
        )

    def fit(self, X_train, y_train):
        self.gb_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.gb_model.predict(X_test)

    def predict_proba(self, X_test):
        return self.gb_model.predict_proba(X_test)
    

# Define GPU XGBoost Regressor Model
class GPUXGBoostModelRegression(nn.Module):
    def __init__(self, max_depth=7, learning_rate=0.1, n_estimators=100):
        super(GPUXGBoostModelRegression, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.gb_model = xgb.XGBRegressor(
            device=self.device,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            tree_method='auto',
            objective='reg:squarederror',
        )

    def fit(self, X_train, y_train):
        self.gb_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.gb_model.predict(X_test)


# Define Gradient Boosting Classifier Model
class GradientBoostingModelBC(nn.Module):
    def __init__(self, max_depth=7, learning_rate=0.1, n_estimators=100):
        super(GradientBoostingModelBC, self).__init__()
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.gb_model = GradientBoostingClassifier(max_depth=self.max_depth, 
                                                   learning_rate=self.learning_rate, 
                                                   n_estimators=self.n_estimators)

    def fit(self, X_train, y_train):
        self.gb_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.gb_model.predict(X_test)

    def predict_proba(self, X_test):
        return self.gb_model.predict_proba(X_test)


# Define Gradient Boosting Regressor Model
class GradientBoostingModelRegression(nn.Module):
    def __init__(self, max_depth=7, learning_rate=0.1, n_estimators=100):
        super(GradientBoostingModelRegression, self).__init__()
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.gb_model = GradientBoostingRegressor(max_depth=self.max_depth, 
                                                  learning_rate=self.learning_rate, 
                                                  n_estimators=self.n_estimators)

    def fit(self, X_train, y_train):
        self.gb_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.gb_model.predict(X_test)


# Define GPU Random Forest models using XGBoost
class GPURandomForestModelBC(nn.Module):
    def __init__(self, n_estimators=512, max_depth=None, learning_rate=1.0):
        super(GPURandomForestModelBC, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rf_model = xgb.XGBRFClassifier(
            device=self.device,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            tree_method='auto',
            objective='binary:hinge',
            learning_rate=learning_rate,
        )

    def fit(self, X_train, y_train):
        self.rf_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.rf_model.predict(X_test)

    def predict_proba(self, X_test):
        return self.rf_model.predict_proba(X_test)
    

class GPURandomForestModelRegression(nn.Module):
    def __init__(self, n_estimators=512, max_depth=None, learning_rate=1.0):
        super(GPURandomForestModelRegression, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rf_model = xgb.XGBRFRegressor(
            device=self.device,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            tree_method='auto',
            objective='reg:squarederror',
            learning_rate=learning_rate,
        )

    def fit(self, X_train, y_train):
        self.rf_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.rf_model.predict(X_test)


# Define Random Forest models
class RandomForestModelBC(nn.Module):
    def __init__(self, n_estimators=512, max_depth=None):
        super(RandomForestModelBC, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rf_model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)

    def forward(self, x):
        # Random Forest model doesn't have a forward pass in the traditional sense,
        # so we won't implement this method
        raise NotImplementedError

    def fit(self, X_train, y_train):
        self.rf_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.rf_model.predict(X_test)
    

class RandomForestModelRegression(nn.Module):
    def __init__(self, n_estimators=512, max_depth=None):
        super(RandomForestModelRegression, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)
    
    def forward(self, x):
        # Random Forest model doesn't have a forward pass in the traditional sense,
        # so we won't implement this method
        raise NotImplementedError
    
    def fit(self, X_train, y_train):
        self.rf_model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.rf_model.predict(X_test)


# Define Support Vector Machine models
class SVMModelBC(nn.Module):
    def __init__(self, C=1.0):
        super(SVMModelBC, self).__init__()
        self.C = C
        self.svm_model = LinearSVC(C=self.C)

    def forward(self, x):
        # SVM model doesn't have a forward pass in the traditional sense,
        # so we won't implement this method
        raise NotImplementedError

    def fit(self, X_train, y_train):
        self.svm_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.svm_model.predict(X_test)
    

class SVMModelRegression(nn.Module):
    def __init__(self, C=1.0, kernel='linear', degree=3, cache_size=200):
        super(SVMModelRegression, self).__init__()
        self.C = C
        self.svm_model = LinearSVR(C=self.C)

    def forward(self, x):
        # SVM model doesn't have a forward pass in the traditional sense,
        # so we won't implement this method
        raise NotImplementedError
    
    def fit(self, X_train, y_train):
        self.svm_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.svm_model.predict(X_test)


# Define K-Nearest Neighbors model
class KNNModelBC(nn.Module):
    def __init__(self, n_neighbors=4, p=2, weights='uniform', algorithm='auto', metric='minkowski'):
        super(KNNModelBC, self).__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.metric = metric
        self.knn_model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p,
        )

    def forward(self, x):
        # KNN model doesn't have a forward pass in the traditional sense,
        # so we won't implement this method
        raise NotImplementedError

    def fit(self, X_train, y_train):
        self.knn_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.knn_model.predict(X_test)
    

class KNNModelRegression(nn.Module):
    def __init__(self, n_neighbors=4, p=2, weights='uniform', algorithm='auto', metric='minkowski'):
        super(KNNModelRegression, self).__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.metric = metric
        self.knn_model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            metric=self.metric,
            p=self.p,
            )

    def forward(self, x):
        # KNN model doesn't have a forward pass in the traditional sense,
        # so we won't implement this method
        raise NotImplementedError

    def fit(self, X_train, y_train):
        self.knn_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.knn_model.predict(X_test)
    

class SNNModelBC(nn.Module):
    def __init__(self, input_size, hidden_size=256, dropout=0.8, learn_rate=0.0002):
        super(SNNModelBC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.loss_fxn = nn.BCELoss()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.train_loss_over_time = []
        self.tune_loss_over_time = []
        self.train_accuracy_over_time = []
        self.tune_accuracy_over_time = []
        
        # Layers
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, device=self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=1, device=self.device)
        self.dropout = nn.Dropout(self.dropout)
        self.sigmoid=nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    def forward(self, x):
        out = self.fc1(x) # Input layer
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) # Hidden layer
        out = self.sigmoid(out) # Binary classification
        #out = torch.flatten(out)
        return out
    
    def fit(self, train_dataloader, tune_dataloader=None, epochs=300):
        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            tune_loss = 0.0
            tune_correct = 0
            tune_total = 0

            self.train()
            for x_batch, y_batch in train_dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self(x_batch)
                y_pred = y_pred.to(self.device)
                loss = self.loss_fxn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_total += y_batch.size(0)
                train_correct += (torch.round(y_pred) == y_batch).sum().item() # Have to round the output to get 0 or 1

            epoch_train_loss = train_loss / len(train_dataloader)
            epoch_train_acc = 100 * train_correct / train_total   
            self.train_loss_over_time.append(epoch_train_loss)
            self.train_accuracy_over_time.append(epoch_train_acc)

            if (epoch) % 50 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%')
            
            if tune_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for x_batch, y_batch in tune_dataloader:
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self(x_batch)
                        y_pred = y_pred.to(self.device)
                        loss = self.loss_fxn(y_pred, y_batch)

                        tune_loss += loss.item()
                        tune_total += y_batch.size(0)
                        tune_correct += (torch.round(y_pred) == y_batch).sum().item()
                epoch_tune_loss = tune_loss / len(tune_dataloader)
                epoch_tune_acc = 100 * tune_correct / tune_total
                self.tune_loss_over_time.append(epoch_tune_loss)
                self.tune_accuracy_over_time.append(epoch_tune_acc)

                if (epoch) % 50 == 0:
                    print(f'Epoch [{epoch}/{epochs}], Tune Loss: {epoch_tune_loss:.4f}, Tune Accuracy: {epoch_tune_acc:.2f}%')
            

    def plot_loss(self, output_file=None):
        # Plot training loss over time
        plt.plot(range(len(self.train_loss_over_time)), self.train_loss_over_time, label='Training Loss')
        plt.title('Training Loss over Time')
        # Plot tuning loss over time
        if len(self.tune_loss_over_time) > 0:
            plt.plot(range(len(self.tune_loss_over_time)), self.tune_loss_over_time, label='Tune Loss')
            plt.title('Training and Tuning Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def plot_accuracy(self, output_file=None):
        # Plot training accuracy over time
        plt.plot(range(len(self.train_accuracy_over_time)), self.train_accuracy_over_time, label='Training Accuracy')
        plt.title('Training Accuracy over Time')
        # Plot tuning accuracy over time
        if len(self.tune_accuracy_over_time) > 0:
            plt.plot(range(len(self.tune_accuracy_over_time)), self.tune_accuracy_over_time, label='Tune Accuracy')
            plt.title('Training and Tuning Accuracy over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def predict(self, X):
        # Move X to the device
        X = X.to(self.device)
        self.eval()
        with torch.no_grad():
            print(self.device)
            print(X.device)
            y_pred = self(X)
            y_pred = torch.round(y_pred)
        return y_pred
    

class SNNModelRegression(nn.Module):
    def __init__(self, input_size, hidden_size=256, dropout=0.8, learn_rate=0.0002):
        super(SNNModelRegression, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.loss_fxn = nn.MSELoss()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.train_loss_over_time = []
        self.tune_loss_over_time = []
        
        # Layers
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, device=self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=1, device=self.device)
        self.dropout = nn.Dropout(self.dropout)

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)

    def forward(self, x):
        out = self.fc1(x) # Input layer
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) # Hidden layer
        return out
    
    def fit(self, train_dataloader, tune_dataloader=None, epochs=300):
        for epoch in range(epochs):
            train_loss = 0.0
            tune_loss = 0.0

            self.train()
            for x_batch, y_batch in train_dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self(x_batch)
                y_pred = y_pred.to(self.device)
                loss = self.loss_fxn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            epoch_train_loss = train_loss / len(train_dataloader)
            self.train_loss_over_time.append(epoch_train_loss)

            if (epoch) % 50 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_train_loss:.4f}')
            
            if tune_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for x_batch, y_batch in tune_dataloader:
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self(x_batch)
                        y_pred = y_pred.to(self.device)
                        loss = self.loss_fxn(y_pred, y_batch)

                        tune_loss += loss.item()
                epoch_tune_loss = tune_loss / len(tune_dataloader)
                self.tune_loss_over_time.append(epoch_tune_loss)

                if (epoch) % 50 == 0:
                    print(f'Epoch [{epoch}/{epochs}], Tune Loss: {epoch_tune_loss:.4f}')
            
    def plot_loss(self, output_file=None):
        # Plot training loss over time
        plt.plot(range(len(self.train_loss_over_time)), self.train_loss_over_time, label='Training Loss')
        plt.title('Training Loss over Time')
        # Plot tuning loss over time
        if len(self.tune_loss_over_time) > 0:
            plt.plot(range(len(self.tune_loss_over_time)), self.tune_loss_over_time, label='Tune Loss')
            plt.title('Training and Tuning Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def predict(self, X):
        # Move X to the device
        X = X.to(self.device)
        self.eval()
        with torch.no_grad():
            print(self.device)
            print(X.device)
            y_pred = self(X)
            y_pred = torch.round(y_pred)
        return y_pred
    

class SYNDEEPModelBC(nn.Module):
    def __init__(self, input_size):
        super(SYNDEEPModelBC, self).__init__()
        self.input_size = input_size
        self.dropout = 0.8
        self.loss_fxn = nn.BCELoss()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.train_loss_over_time = []
        self.tune_loss_over_time = []
        self.train_accuracy_over_time = []
        self.tune_accuracy_over_time = []
        
        # Layers
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.input_size, device=self.device)
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=512, device=self.device)
        self.batchnorm1 = nn.BatchNorm1d(num_features=512, device=self.device)
        self.fc2 = nn.Linear(in_features=512, out_features=128, device=self.device)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128, device=self.device)
        self.fc3 = nn.Linear(in_features=128, out_features=32, device=self.device)
        self.batchnorm3 = nn.BatchNorm1d(num_features=32, device=self.device)
        self.fc4 = nn.Linear(in_features=32, out_features=1, device=self.device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.sigmoid=nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, x):
        out = self.batchnorm0(x)
        out = self.fc1(x) # Input layer
        out = self.relu(out)
        out = self.batchnorm1(out)
        out = self.dropout(out)
        out = self.fc2(out) # Hidden layer
        out = self.relu(out)
        out = self.batchnorm2(out)
        out = self.dropout(out)
        out = self.fc3(out) # Hidden layer
        out = self.relu(out)
        out = self.batchnorm3(out)
        out = self.dropout(out)
        out = self.fc4(out) # Hidden layer
        out = self.sigmoid(out) # Binary classification
        return out
    
    def fit(self, train_dataloader, tune_dataloader=None, epochs=300):
        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            tune_loss = 0.0
            tune_correct = 0
            tune_total = 0

            self.train()
            for x_batch, y_batch in train_dataloader:
                # if the batch size is 1, the model will throw an error, so just skip it
                if x_batch.size(0) == 1:
                    continue
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self(x_batch)
                y_pred = y_pred.to(self.device) # Do you need this at this point?
                loss = self.loss_fxn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_total += y_batch.size(0)
                train_correct += (torch.round(y_pred) == y_batch).sum().item() # Have to round the output to get 0 or 1

            epoch_train_loss = train_loss / len(train_dataloader)
            epoch_train_acc = 100 * train_correct / train_total
            self.train_loss_over_time.append(epoch_train_loss)
            self.train_accuracy_over_time.append(epoch_train_acc)
            
            if (epoch) % 50 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%')
            
            if tune_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for x_batch, y_batch in tune_dataloader:
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self(x_batch)
                        y_pred = y_pred.to(self.device) # Do you need this at this point?
                        loss = self.loss_fxn(y_pred, y_batch)

                        tune_loss += loss.item()
                        tune_total += y_batch.size(0)
                        tune_correct += (torch.round(y_pred) == y_batch).sum().item()
                epoch_tune_loss = tune_loss / len(tune_dataloader)
                epoch_tune_acc = 100 * tune_correct / tune_total
                self.tune_loss_over_time.append(epoch_tune_loss)
                self.tune_accuracy_over_time.append(epoch_tune_acc)

                if (epoch) % 50 == 0:
                    print(f'Epoch [{epoch}/{epochs}], Tune Loss: {epoch_tune_loss:.4f}, Tune Accuracy: {epoch_tune_acc:.2f}%')
                 
    def plot_loss(self, output_file=None):
        # Plot training loss over time
        plt.plot(range(len(self.train_loss_over_time)), self.train_loss_over_time, label='Training Loss')
        plt.title('Training Loss over Time')
        # Plot tuning loss over time
        if len(self.tune_loss_over_time) > 0:
            plt.plot(range(len(self.tune_loss_over_time)), self.tune_loss_over_time, label='Tune Loss')
            plt.title('Training and Tuning Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def plot_accuracy(self, output_file=None):
        # Plot training accuracy over time
        plt.plot(range(len(self.train_accuracy_over_time)), self.train_accuracy_over_time, label='Training Accuracy')
        plt.title('Training Accuracy over Time')
        # Plot tuning accuracy over time
        if len(self.tune_accuracy_over_time) > 0:
            plt.plot(range(len(self.tune_accuracy_over_time)), self.tune_accuracy_over_time, label='Tune Accuracy')
            plt.title('Training and Tuning Accuracy over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def predict(self, X):
        # Move X to the device
        X = X.to(self.device)
        self.eval()
        with torch.no_grad():
            print(self.device)
            print(X.device)
            y_pred = self(X)
            y_pred = torch.round(y_pred)
        return y_pred
    

class SYNDEEPModelRegression(nn.Module):

    def __init__(self, input_size):
        super(SYNDEEPModelRegression, self).__init__()
        self.input_size = input_size
        self.dropout = 0.8
        self.loss_fxn = nn.MSELoss()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.train_loss_over_time = []
        self.tune_loss_over_time = []
        
        # Layers
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.input_size, device=self.device)
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=512, device=self.device)
        self.batchnorm1 = nn.BatchNorm1d(num_features=512, device=self.device)
        self.fc2 = nn.Linear(in_features=512, out_features=128, device=self.device)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128, device=self.device)
        self.fc3 = nn.Linear(in_features=128, out_features=32, device=self.device)
        self.batchnorm3 = nn.BatchNorm1d(num_features=32, device=self.device)
        self.fc4 = nn.Linear(in_features=32, out_features=1, device=self.device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

    def forward(self, x):
        out = self.batchnorm0(x)
        out = self.fc1(x) # Input layer
        out = self.relu(out)
        out = self.batchnorm1(out)
        out = self.dropout(out)
        out = self.fc2(out) # Hidden layer
        out = self.relu(out)
        out = self.batchnorm2(out)
        out = self.dropout(out)
        out = self.fc3(out) # Hidden layer
        out = self.relu(out)
        out = self.batchnorm3(out)
        out = self.dropout(out)
        out = self.fc4(out) # Hidden layer
        return out
    
    def fit(self, train_dataloader, tune_dataloader=None, epochs=300):
        for epoch in range(epochs):
            train_loss = 0.0
            tune_loss = 0.0

            self.train()
            for x_batch, y_batch in train_dataloader:
                # if the batch size is 1, the model will throw an error, so just skip it
                if x_batch.size(0) == 1:
                    continue
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self(x_batch)
                y_pred = y_pred.to(self.device)
                loss = self.loss_fxn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            epoch_train_loss = train_loss / len(train_dataloader)
            self.train_loss_over_time.append(epoch_train_loss)
            
            if (epoch) % 50 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_train_loss:.4f}')

            if tune_dataloader is not None:
                self.eval()
                with torch.no_grad():
                    for x_batch, y_batch in tune_dataloader:
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self(x_batch)
                        y_pred = y_pred.to(self.device)
                        loss = self.loss_fxn(y_pred, y_batch)

                    tune_loss += loss.item()
                epoch_tune_loss = tune_loss / len(tune_dataloader)
                self.tune_loss_over_time.append(epoch_tune_loss)

                if (epoch) % 50 == 0:
                    print(f'Epoch [{epoch}/{epochs}], Tune Loss: {epoch_tune_loss:.4f}')

    def plot_loss(self, output_file=None):
        # Plot training loss over time
        plt.plot(range(len(self.train_loss_over_time)), self.train_loss_over_time, label='Training Loss')
        plt.title('Training Loss over Time')
        # Plot tuning loss over time
        if len(self.tune_loss_over_time) > 0:
            plt.plot(range(len(self.tune_loss_over_time)), self.tune_loss_over_time, label='Tune Loss')
            plt.title('Training and Tuning Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()

    def predict(self, X):
        # Move X to the device
        X = X.to(self.device)
        self.eval()
        with torch.no_grad():
            print(self.device)
            print(X.device)
            y_pred = self(X)
            y_pred = torch.round(y_pred)
        return y_pred