using Plots


function createNeuralNetwork(n_input, n_hidden, n_output)
    Dict("W1" => rand(n_input, n_hidden), 
         "b1" => zeros(1, n_hidden),
         "W2" => rand(n_hidden, n_output),
         "b2" => zeros(1, n_output)
        )
end

function ReLU(x)
    x > 0 ? x : 0
end


function mean_squared_error(y, t)
    sum((y - t).^ 2) / 2
end

function soft_max(a)
    exp_a = exp.(a .- maximum(a)) 
    exp_a / sum(exp_a)
end

function numerical_diff(f, x)
    h = 1e-4 # 0.0001
    (f(x+h) - f(x-h)) / 2*h
end

sigmoid(x) = 1 / (1 + exp(-x))

function predict(nl::Dict, x)
    s1 = ReLU.(x * nl["W1"] + nl["b1"])
    s1 * nl["W2"] + nl["b2"]
end

function loss(nl::Dict, x, t)
    pd = predict(nl, x)
    mean_squared_error(pd, t)
end

function numerical_gradient(nl::Dict, f, layer::String)
    h = 1e-4

    grad = zeros(size(nl[layer]))

    tnl = copy(nl)
    for i in eachindex(nl[layer])
        tnl[layer][i] = nl[layer][i] + h
        fxh1 = f(tnl)

        tnl[layer][i] = nl[layer][i] - h
        fxh2 = f(tnl)

        grad[i] = (fxh1 - fxh2) / (2*h)

        tnl[layer][i] = nl[layer][i]
    end

    return grad
end


#----------------
#----------------

train_data = [
              ([0 0],[1]),
              ([0 1],[0]),
              ([1 0],[0]),
              ([1 1],[1])
             ]

learning_rate = 0.1
nl = createNeuralNetwork(2,4,1)

global y1 = []
global y2 = []
global y3 = []
global y4 = []

train_count = 1000

for i in 1:train_count
    gr = copy(nl)
    # learning
    function mean_loss(_nl::Dict)
        ps = []
        as = []
        for d in train_data
            append!(ps, predict(_nl, d[1]))
            append!(as, d[2])
        end
        mean_squared_error(ps, as)
    end
    for k in keys(nl)
        gr[k] = learning_rate * numerical_gradient(nl, mean_loss, k)
    end
    for k in keys(nl)
        nl[k] -= gr[k]
    end

    append!(y1, predict(nl, [0 0]))
    append!(y2, predict(nl, [1 0]))
    append!(y3, predict(nl, [0 1]))
    append!(y4, predict(nl, [1 1]))
end

plot(1:train_count, y1, label="[0 0]")
plot!(1:train_count, y2, label="[1 0]")
plot!(1:train_count, y3, label="[0 1]")
p = plot!(1:train_count, y4, label="[1 1]")

display(p)

while true
end


