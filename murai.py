def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    """Train and evaluate a model with CPU."""
    
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        X = train_inter
        prevY = Y0
        converged = False
        
        with autograd.record():
            W = f(X)
            
            while not converged:
                Y = invA @ (W @ prevY + Y0)
                
                if np.linalg.norm(prevY-Y) < eps:
                    converged = True
                prevY = Y
            
            l = loss(Y,Ytrain)
        
        l.backward()
        
        if trainer is None:
            sgd(params, lr, batch_size) # editar
        
        else:
            trainer.step(batch_size)
        
        Y = Y.astype('float32')
        train_l_sum = l.asscalar()
        train_acc_sum = ( (-1+2*(Y>0)) == Ytrain).sum().asscalar()
        n = len(Ytrain)
        test_acc = evaluate_accuracy(test_iter, net) # editar
        print('epoch {}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}'.format(epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
