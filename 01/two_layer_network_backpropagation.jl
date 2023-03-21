using Plots


struct Affine
    X::Matrix
    W::Matrix
    B::Matrix
    dW::Matrix
    dB::Matrix
end

#-------
# ReLU
#-------
function Affine(n_input, n_output)
    Affine(
         zeros(1, n_input),         # X
         rand(n_input, n_output),   # W
         zeros(1, n_output),        # B
         zeros(n_input, n_output),  # dW
         zeros(1, n_output),        # dB
        )
end

function forward!(a::Affine, x)
    copy!(a.X, x)
    y = a.X * a.W + a.B
    return y
end

function backward!(a::Affine, dout)
    dX = dout * a.W'
    dW = a.X' * dout
    copyto!(a.dW, dW)
    copyto!(a.dB, dout)
    return dX
end

function update!(a::Affine, learning_rate )
    nw = a.W - learning_rate * a.dW
    nb = a.B - learning_rate * a.dB
    copy!(a.W, nw)
    copy!(a.B, nb)
end


#-------
# ReLU
#-------
struct ReLU
    mask::BitMatrix
end

function ReLU(n)
    ReLU(falses(1, n))
end

function forward!(r::ReLU, x)
    mask = x .<= 0
    copy!(r.mask, mask)
    x[mask] .= 0 
    return x
end

function backward!(r::ReLU, dout)
    dout[r.mask] .= 0
    return dout
end

function update!(r::ReLU, learning_rate)
    return r
end

#--------
#--------

function predict!(nl, d)
    for e in nl
        d = forward!(e, d)
    end
    return d
end

function training!(nl, loss)
    l = loss
    for e in reverse(nl)
        l = backward!(e, l)
    end
end

function update_weight!(nl, learning_rate)
    for e in nl
        update!(e, learning_rate)
    end
end


# 3 layer neural network
nl = [Affine(2,3),
      ReLU(3),
      Affine(3,1)]

train_data = [
              ([0 0],[1]),
              ([0 1],[0]),
              ([1 0],[0]),
              ([1 1],[1])
             ]

learning_rate = 0.05
train_count = 500

global y1 = []
global y2 = []
global y3 = []
global y4 = []
global loss = []
for i in 1:train_count
    for d in train_data
        y = predict!(nl, d[1])
        training!(nl, y - d[2]) 
        update_weight!(nl, learning_rate)
    end

    append!(y1, predict!(nl, [0 0]))
    append!(y2, predict!(nl, [0 1]))
    append!(y3, predict!(nl, [1 0]))
    append!(y4, predict!(nl, [1 1]))
end

plot(1:train_count, y1, label="[0 0]")
plot!(1:train_count, y2, label="[0 1]")
plot!(1:train_count, y3, label="[1 0]")
p = plot!(1:train_count, y4, label="[1 1]")

display(p)

while true
end

