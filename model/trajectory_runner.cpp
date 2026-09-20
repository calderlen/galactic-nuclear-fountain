#include "csv.h"
#include "orbits.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <zlib.h>

namespace {

constexpr double kms_to_kpc_per_Myr = 1.022712165e-3;

double finite_number(const std::string& text, const std::string& label) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("Expected a finite number for " + label + ": " + text);
    }
    return value;
}

std::uint64_t unsigned_number(const std::string& text, const std::string& label) {
    std::uint64_t value;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument("Expected an unsigned integer for " + label + ": " + text);
    }
    return value;
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else if (character != '\r' && character != '\n') {
            field.push_back(character);
        }
    }
    if (quoted) throw std::runtime_error("Unterminated quoted CSV field");
    fields.push_back(field);
    return fields;
}

class GzipInput {
public:
    explicit GzipInput(const std::filesystem::path& path) : file_(gzopen(path.string().c_str(), "rb")) {
        if (file_ == nullptr) throw std::runtime_error("Could not read " + path.string());
    }
    ~GzipInput() { gzclose(file_); }
    GzipInput(const GzipInput&) = delete;
    GzipInput& operator=(const GzipInput&) = delete;

    bool getline(std::string& line) {
        line.clear();
        char buffer[65536];
        while (true) {
            char* result = gzgets(file_, buffer, static_cast<int>(sizeof(buffer)));
            if (result == nullptr) {
                if (gzeof(file_)) return !line.empty();
                int error_number = Z_OK;
                const char* message = gzerror(file_, &error_number);
                throw std::runtime_error(std::string("Could not read orbit catalog: ") + message);
            }
            line += buffer;
            if (!line.empty() && line.back() == '\n') return true;
        }
    }

private:
    gzFile file_;
};

struct LaunchRecord {
    std::uint64_t parcel_id;
    double R0, z0, vR0_kms, vz0_kms, j0;
    double v_k_kms, theta_rad, psi_rad;
    std::string saved_status;
};

const char* status_name(OrbitStatus status) {
    switch (status) {
        case OrbitStatus::Returned: return "returned";
        case OrbitStatus::TimedOut: return "timed_out";
        case OrbitStatus::IntegrationFailed: return "integration_failed";
    }
    throw std::logic_error("Unknown orbit status");
}

std::unordered_map<std::string, std::string> read_settings(const std::filesystem::path& path) {
    namespace csv = galactic_nuclear_fountain::csv;
    const csv::Table table = csv::read(path);
    if (table.header.size() < 2) throw std::invalid_argument("Configuration needs at least two columns");
    const std::string key_column = table.header[0];
    const std::string value_column = table.header[1];
    std::unordered_map<std::string, std::string> settings;
    for (const auto& row : table.rows) {
        settings[csv::value(row, key_column)] = csv::value(row, value_column);
    }
    return settings;
}

double setting(const std::unordered_map<std::string, std::string>& settings,
               const std::string& name) {
    return finite_number(settings.at(name), name);
}

double optional_setting(const std::unordered_map<std::string, std::string>& settings,
                        const std::string& name, double fallback) {
    const auto found = settings.find(name);
    return found == settings.end() ? fallback : finite_number(found->second, name);
}

std::vector<LaunchRecord> read_launches(const std::filesystem::path& path,
                                        const std::vector<std::uint64_t>& requested,
                                        const OrbitParameters& potential) {
    GzipInput input(path);
    std::string line;
    if (!input.getline(line)) throw std::runtime_error("Orbit catalog is empty");
    const std::vector<std::string> header = parse_csv_line(line);
    std::unordered_map<std::string, std::size_t> column;
    for (std::size_t index = 0; index < header.size(); ++index) column[header[index]] = index;
    const auto require = [&](const std::string& name) {
        if (!column.contains(name)) throw std::invalid_argument("Orbit catalog needs column " + name);
    };
    for (const char* name : {"parcel_id", "R0_kpc", "z0_kpc", "vR0_kms", "vz0_kms", "status"}) {
        require(name);
    }
    const std::string j_column = column.contains("j_z0_kpc2_per_Myr")
        ? "j_z0_kpc2_per_Myr" : "j_z_kpc2_per_Myr";
    require(j_column);

    const std::unordered_set<std::uint64_t> wanted(requested.begin(), requested.end());
    std::unordered_map<std::uint64_t, LaunchRecord> found;
    while (found.size() < wanted.size() && input.getline(line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = parse_csv_line(line);
        if (fields.size() < header.size()) continue;
        const auto text = [&](const std::string& name) -> const std::string& {
            return fields.at(column.at(name));
        };
        const std::uint64_t parcel_id = unsigned_number(text("parcel_id"), "parcel_id");
        if (!wanted.contains(parcel_id)) continue;
        LaunchRecord record{
            parcel_id,
            finite_number(text("R0_kpc"), "R0_kpc"),
            finite_number(text("z0_kpc"), "z0_kpc"),
            finite_number(text("vR0_kms"), "vR0_kms"),
            finite_number(text("vz0_kms"), "vz0_kms"),
            finite_number(text(j_column), j_column),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            text("status")
        };
        if (column.contains("v_k_kms")) record.v_k_kms = finite_number(text("v_k_kms"), "v_k_kms");
        if (column.contains("theta_rad")) record.theta_rad = finite_number(text("theta_rad"), "theta_rad");
        if (column.contains("psi_rad")) record.psi_rad = finite_number(text("psi_rad"), "psi_rad");
        if (!std::isfinite(record.v_k_kms) || !std::isfinite(record.theta_rad) || !std::isfinite(record.psi_rad)) {
            const double vphi0_kms = record.j0 / record.R0 / kms_to_kpc_per_Myr;
            const double vc0_kms = circular_velocity(record.R0, potential) / kms_to_kpc_per_Myr;
            const double dvphi_kms = vphi0_kms - vc0_kms;
            record.v_k_kms = std::hypot(record.vR0_kms, record.vz0_kms, dvphi_kms);
            record.theta_rad = record.v_k_kms == 0.0 ? 0.0
                : std::acos(std::clamp(record.vz0_kms / record.v_k_kms, -1.0, 1.0));
            record.psi_rad = std::atan2(dvphi_kms, record.vR0_kms);
            if (record.psi_rad < 0.0) record.psi_rad += 2.0 * std::numbers::pi;
        }
        found.emplace(parcel_id, std::move(record));
    }

    std::vector<LaunchRecord> launches;
    for (const std::uint64_t parcel_id : requested) {
        const auto match = found.find(parcel_id);
        if (match == found.end()) {
            throw std::runtime_error("Parcel " + std::to_string(parcel_id) +
                                     " was not found in the available orbit catalog");
        }
        launches.push_back(match->second);
    }
    return launches;
}

}

int main(int argc, char* argv[]) {
    const bool help = argc >= 2 && std::string(argv[1]) == "--help";
    if (argc < 6 || help) {
        std::cout << "Usage: " << argv[0]
                  << " CONFIG.csv ORBITS.csv[.gz] OUTPUT.csv SAMPLE_DT_Myr PARCEL_ID [PARCEL_ID ...]\n"
                  << "Re-integrates selected saved launches and samples t,R,z,vR,vz,vphi,j_z,m.\n"
                  << "The configuration may be a current config.csv or an older parameters.csv.\n";
        return help ? 0 : 1;
    }
    try {
        const std::filesystem::path config_path = argv[1];
        const std::filesystem::path catalog_path = argv[2];
        const std::filesystem::path output_path = argv[3];
        const double output_dt = finite_number(argv[4], "SAMPLE_DT_Myr");

        std::vector<std::uint64_t> requested;
        std::unordered_set<std::uint64_t> unique;
        for (int index = 5; index < argc; ++index) {
            const std::uint64_t parcel_id = unsigned_number(argv[index], "PARCEL_ID");
            if (!unique.insert(parcel_id).second) {
                throw std::invalid_argument("Duplicate parcel ID: " + std::to_string(parcel_id));
            }
            requested.push_back(parcel_id);
        }

        const auto settings = read_settings(config_path);
        const OrbitParameters potential{
            .M_d = setting(settings, "M_d_Msun"),
            .a_d = setting(settings, "a_d_kpc"),
            .b_d = setting(settings, "b_d_kpc"),
            .M_b = setting(settings, "M_b_Msun"),
            .a_b = setting(settings, "a_b_kpc"),
            .rho_s = setting(settings, "rho_s_Msun_kpc3"),
            .r_s = setting(settings, "r_s_kpc"),
            .rho_CGM = optional_setting(settings, "rho_CGM_Msun_kpc3", 0.0),
            .sigma = optional_setting(settings, "parcel_sigma_kpc2", 0.0),
            .C_D = optional_setting(settings, "C_D", 0.0),
            .epsilon = optional_setting(settings, "epsilon", 0.0),
            .beta = optional_setting(settings, "beta", 0.0)
        };
        const double m0 = optional_setting(settings, "parcel_m0_Msun", 1.0);
        const double h_0 = setting(settings, "h_0_Myr");
        const double h_max = setting(settings, "h_max_Myr");
        const double atol = setting(settings, "atol");
        const double rtol = setting(settings, "rtol");
        const double t_stop = setting(settings, "t_stop_Myr");
        const std::vector<LaunchRecord> launches = read_launches(catalog_path, requested, potential);

        if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
        std::ofstream output(output_path);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output << std::setprecision(17)
               << "parcel_id,status,v_k_kms,theta_rad,psi_rad,t_Myr,R_kpc,z_kpc,"
                  "vR_kms,vz_kms,vphi_kms,j_z_kpc2_per_Myr,m_Msun\n";
        for (const LaunchRecord& launch : launches) {
            const DoubleVec initial{
                launch.R0, launch.z0,
                launch.vR0_kms * kms_to_kpc_per_Myr,
                launch.vz0_kms * kms_to_kpc_per_Myr,
                launch.j0, m0
            };
            const OrbitTrace trace = integrate_orbit_trace(
                initial, potential, h_0, atol, rtol, t_stop, output_dt, h_max);
            const std::string status = status_name(trace.result.status);
            if (status != launch.saved_status) {
                throw std::runtime_error("Re-integrated status for parcel " +
                                         std::to_string(launch.parcel_id) + " is " + status +
                                         ", but the saved catalog says " + launch.saved_status);
            }
            for (const OrbitSample& sample : trace.samples) {
                const DoubleVec& state = sample.state;
                output << launch.parcel_id << ',' << status << ',' << launch.v_k_kms << ','
                       << launch.theta_rad << ',' << launch.psi_rad << ',' << sample.t << ','
                       << state[0] << ',' << state[1] << ','
                       << state[2] / kms_to_kpc_per_Myr << ','
                       << state[3] / kms_to_kpc_per_Myr << ','
                       << state[4] / state[0] / kms_to_kpc_per_Myr << ','
                       << state[4] << ',' << state[5] << '\n';
            }
            std::cout << "Parcel " << launch.parcel_id << ": " << status
                      << ", " << trace.samples.size() << " samples\n";
        }
        std::cout << "Wrote " << output_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Trajectory export failed: " << error.what() << '\n';
        return 1;
    }
}
