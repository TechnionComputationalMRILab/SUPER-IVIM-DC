import matplotlib.pyplot as plt


def loss_plot_supervised(loss_Dp, loss_Dt, loss_Fp, loss_train,
                         loss_val, val_loss_Dp, val_loss_Dt, val_loss_Fp,
                         loss_recon=[], val_loss_recon=[]):

    plt.figure(427)
    plt.clf()
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ion()
    plt.legend(["Train", "Validation"])
    plt.title("Total Loss (Reconstruction + IVIM Parameters)")
    plt.show()

    plt.figure(5848)
    plt.clf()
    plt.plot(loss_Dp, 'k')
    plt.plot(val_loss_Dp, 'y')

    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ion()
    plt.legend(["Train loss D*", "Valid loss D*"])
    plt.title("D* Error")
    plt.show()

    plt.figure(146)
    plt.clf()
    plt.plot(loss_Dt, 'c')
    plt.plot(val_loss_Dt, 'm')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ion()
    plt.legend(["Train loss Dt", "Valid loss Dt"])
    plt.title("D Error")
    plt.show()

    plt.figure(319)
    plt.clf()
    plt.plot(loss_Fp, 'g')
    plt.plot(val_loss_Fp, 'r')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ion()
    plt.legend(["Train loss Fp", "Valid loss Fp"])
    plt.title("Fp Error")
    plt.show()

    if val_loss_recon:  # check if the variable is not empty
        plt.figure(213)
        plt.clf()
        plt.plot(loss_recon, 'k')
        plt.plot(val_loss_recon, 'g')
        plt.yscale("log")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ion()
        plt.legend(["Train loss Recon", "Valid loss Recon"])
        plt.title("Reconstruction Error")
        plt.show()