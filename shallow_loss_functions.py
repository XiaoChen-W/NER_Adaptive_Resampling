# -*- coding: utf-8 -*-




def _multinomial_loss(w, X, Y, alpha, sample_weight):
    
    # Self-implemented Focal Loss and Dice Loss functions
    # Use the following code to replace the function in sklearn.linear_model.LogisticRegression with the same name
    # As the original LogisticRegression package is well-encapsulated, we do not set function name as hyperparameters since it will increase working load of users.
    """Computes multinomial loss and class probabilities.

    Parameters
    ----------
    w : ndarray of shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    Y : ndarray of shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,)
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    p : ndarray of shape (n_samples, n_classes)
        Estimated class probabilities.

    w : ndarray of shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    log_p = deepcopy(p)
    p = np.exp(p, p)
    # Focal Loss implementation, comment the following two lines if use Dice Loss
    # focal_weight = (1-p*Y)**2
    # loss = -sample_weight*(Y * focal_weight*log_p).sum()
    # Dice Loss implementation, comment the following two lines if use Focal Loss
    dice_weight =  (2*(p*Y)+1)/((p**2+Y**2)+1)
    loss = (1- dice_weight).sum()
    loss += 0.5 * alpha * squared_norm(w)
    return loss, p, w


def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    
    # Self-implemented derivation of Focal Loss and Dice Loss functions
    # Use the following code to replace the function in sklearn.linear_model.LogisticRegression with the same name
    # As the original LogisticRegression package is well-encapsulated, we do not set function name as hyperparameters since it will increase working load of users.
    """Computes the multinomial loss, gradient and class probabilities.

    Parameters
    ----------
    w : ndarray of shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    Y : ndarray of shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,)
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    grad : ndarray of shape (n_classes * n_features,) or \
            (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    p : ndarray of shape (n_samples, n_classes)
        Estimated class probabilities

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)),
                    dtype=X.dtype)
    sample_weight = sample_weight[:, np.newaxis]
    
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    p_c =  np.array([np.sum(p*Y,axis = 1)])[0]
    
    # See the mathematical derivation process in our technical appendix
    
    # Focal Loss derivation implementation, comment the following three lines if use Dice Loss
    gamma = 2
    p_c = np.array([[i] for i in p_c])+np.zeros(p.shape)
    a_c = -gamma*np.log(p_c)*p_c*(1-p_c)**(gamma-1)+(1-p_c)**gamma
    diff =  a_c*(p-Y)
    # Dice Loss derivation implementation, comment the following eleven lines if use Focal Loss
    # gamma = 1
    # a_c= np.array([2*(1-p_c)*(1+gamma+p_c)*p_c/((gamma+p_c**2+1)**2)])
    # diff =  a_c.T*(p-Y)
    # p_j = p*(1-Y)
    # b_j= 2*gamma*p_j**2/((gamma+p_j**2)**2)
    # for i in range(len(p[0])):
    #     for g in range(len(p[0])):
    #         if i == g:
    #             diff[:,i:i+1] -= b_j[:,g:g+1]*(p[:,i:i+1]-1)
    #         else:
    #             diff[:,i:i+1] -= b_j[:,g:g+1]*(p[:,i:i+1])            
                
        
    grad[:, :n_features] = np.dot(X.T,diff).T 
    grad[:, :n_features] += alpha * w   
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p

