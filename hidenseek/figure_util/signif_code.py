def signif_code(p):
    """
    significance code         p-value
       ***                 [0, 0.001]
        **              (0.001, 0.01]
         *               (0.01, 0.05]
         .                (0.05, 0.1]
                             (0.1, 1]
    """
    if p <= 0.001:
        return '***'
    elif 0.001 < p <= 0.01:
        return '**'
    elif 0.01 < p <= 0.05:
        return '*'
    else:
        return ''

