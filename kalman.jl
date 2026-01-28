using Plots
using LinearAlgebra
using DSP
using Rotations

# Ψ/θ/φ relative to NED reference frame.

# state: [lat, long, alt, vn, ve, vd, Ψ, θ, φ]
# inputs: [lat, long, alt_gps, alt_pres, ax, ay, az, p, q, r, Ψ, θ, φ]
L = 9  # state dimension
M = 13  # input dimension

xlatᵢ, xlongᵢ, xaltᵢ, xvnᵢ, xveᵢ, xvdᵢ, xΨᵢ, xθᵢ, xφᵢ = 1:L
ulatᵢ, ulongᵢ, ualtgᵢ, ualtpᵢ, uaxᵢ, uayᵢ, uazᵢ, upᵢ, uqᵢ, urᵢ = 1:M

m_to_deg = π / 180 * 6371000  # meters per degree of latitude

function f(x̂, u, dt)
     lat, long, alt, vn, ve, vd, Ψ, θ, φ = x̂
     ax, ay, az, p, q, r = u[5:end]

     # transform accelerations from body frame to NED frame
     R_bn = RotZYX(Ψ, θ, φ)
     an, ae, ad = R_bn * [ax, ay, az]

     # transform angular rates from body frame to NED frame
     Ψ_rate, θ_rate, φ_rate = [
          1 sin(Ψ) * tan(θ) cos(Ψ) * tan(θ);
          0 cos(Ψ) -sin(Ψ);
          0 sin(Ψ) / cos(θ) cos(Ψ) / cos(θ);
     ] * [p, q, r]

     return [
          lat + (vn * dt) / m_to_deg,
          long + (ve * dt) / (m_to_deg * cosd(lat)),
          alt - vd * dt,
          vn + an * dt,
          ve + ae * dt,
          vd + ad * dt,
          Ψ + Ψ_rate * dt,
          θ + θ_rate * dt,
          φ + φ_rate * dt,
     ]
end

function jacobian_h(x̂)
     F = zeros(L, L)
     F[xlatᵢ, xlatᵢ] = 1
     F[xlatᵢ, xvnᵢ] = 1 / m_to_deg

     F[xlongᵢ, xlongᵢ] = 1
     F[xlongᵢ, xveᵢ] = 1 / (m_to_deg * cosd(x̂[xlatᵢ]))

     F[xaltᵢ, xvdᵢ] = -1
     F[xaltᵢ, xaltᵢ] = 1

     # what the fuck (file:///home/kyle/Downloads/Beard,%20Randal%20W._McLain,%20Timothy%20W%20-%20Small%20Unmanned%20Aircraft_%20Theory%20and%20Practice%20(2012,%20Princeton%20University%20Press)%20-%20libgen.li.pdf)
     F[xvnᵢ, Ψ] = dt * (- ax * sin(Ψ) * cos(θ) - ay * sin(Ψ) * sin(θ) * sin(φ) - ay * cos(Ψ) * cos(φ) - az * sin(Ψ) * sin(θ) * cos(φ) + az * cos(Ψ) * sin(φ))
     F[xvnᵢ, θ] = dt * (- ax * cos(Ψ) * sin(θ) + ay * sin(Ψ) * cos(θ) * sin(φ) + az * cos(Ψ) * cos(θ) * cos(φ))
     F[xvn₁, φ] = dt * (ay * cos(Ψ) * sin(θ) * cos(φ) + ay * sin(Ψ) * sin(φ) - az * cos(Ψ) * sin(θ) * sin(φ) + az * sin(Ψ) * cos(φ))

     F[xveᵢ, Ψ] = dt * (ax * cos(Ψ) * cos(θ) + ay * sin(Ψ) * sin(θ) * sin(φ) - ay * sin(Ψ) * cos(φ) + az * cos(Ψ) * sin(θ) * cos(φ) + az * sin(Ψ) * sin(φ))
     F[xveᵢ, θ] = dt * (- ax * sin(Ψ) * sin(θ) + ay * sin(Ψ) * cos(θ) * sin(φ) + az * sin(Ψ) * cos(θ) * cos(φ))
     F[xveᵢ, φ] = dt * (ay * sin(Ψ) * sin(θ) * cos(φ) - ay * cos(Ψ) * sin(φ) - az * sin(Ψ) * sin(θ) * sin(φ) - az * cos(Ψ) * cos(φ))

     F[xvdᵢ, Ψ] = 0
     F[xvdᵢ, θ] = -dt * ( ax * cos(θ) - ay * sin(θ) * sin(φ) - az * sin(θ) * cos(φ))
     F[xvdᵢ, φ] = dt * (ay * cos(θ) * cos(φ) - az * cos(θ) * sin(φ))


     F[xΨᵢ, xΨᵢ] = 1 + dt * (q * tan(Ψ) * cos(Ψ) - r * sin(Ψ) * (sec(θ)^2))
     F[xΨᵢ, xθᵢ] = dt * (q * sin(Ψ) / (cos(θ)^2) + r * cos(Ψ) * (sec(θ)^2))

     F[xθᵢ, xΨᵢ] = dt * (q * sin(Ψ) + r * cos(Ψ))
     F[xθᵢ, xθᵢ] = 1
     
     F[xφᵢ, xΨᵢ] = dt * (q * cos(Ψ) * tan(θ) - r * sin(Ψ) * tan(θ))
     F[xφᵢ, xθᵢ] = dt * (q * sin(Ψ) * tan(θ) / cos(θ) + r * cos(Ψ) * tan(θ) / cos(θ))
     F[xφᵢ, xφᵢ] = 1

     return F
end

