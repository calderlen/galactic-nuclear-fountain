#pragma once

// Surface mass rates use Msun/Gyr/kpc^2. Integrated mass flux interfaces remain
// Msun/yr, steady velocities remain kpc/yr, and orbital/evolution clocks use Myr.
namespace units {
inline constexpr double years_per_gyr = 1e9;
inline constexpr double myr_per_gyr = 1e3;
inline constexpr double per_gyr(double per_year) { return per_year*years_per_gyr; }
inline constexpr double per_year(double per_gyr) { return per_gyr/years_per_gyr; }
inline constexpr double per_myr(double per_gyr) { return per_gyr/myr_per_gyr; }
}
