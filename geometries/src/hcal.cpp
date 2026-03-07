// src/hcal.cpp
#include <DD4hep/DetFactoryHelper.h>
#include <DD4hep/Printout.h>
#include <DD4hep/Shapes.h>
#include <DD4hep/Volumes.h>
#include <XML/XMLElements.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dd4hep;
using namespace dd4hep::detail;

namespace {

constexpr const char* kPluginName = "hcal_plugin";
constexpr int kSegmentCount = 3;

struct GeoParameters {
  double half_width_x = 0.50 * dd4hep::m;
  double half_width_y = 0.30 * dd4hep::m;
  double front_face_z = 0.20 * dd4hep::m;
  double spacer_thickness = 0.0;
  int layer_count = 10;
  std::string side = "-z";
  std::string absorber_material_name;
  std::string active_material_name;
  std::string spacer_material_name;
  std::array<int, kSegmentCount> segment_layer_counts {{0, 0, 0}};
  std::array<double, kSegmentCount> segment_absorber_thicknesses {{0.0, 0.0, 0.0}};
  std::array<double, kSegmentCount> segment_scintillator_thicknesses {{0.0, 0.0, 0.0}};
};

struct ResolvedSegment {
  int layer_count = 0;
  double absorber_thickness = 0.0;
  double spacer_thickness = 0.0;
  double scintillator_thickness = 0.0;
};

struct SegmentVolumes {
  int layer_count = 0;
  double absorber_half_thickness = 0.0;
  double spacer_half_thickness = 0.0;
  double scintillator_half_thickness = 0.0;
  Volume absorber_volume;
  Volume spacer_volume;
  Volume scintillator_volume;
};

bool find_parameter_value(xml_h xml_handle, const char* parameter_name, std::string& value) {
  for (xml_coll_t parameter(xml_handle, _U(parameter)); parameter; ++parameter) {
    xml_comp_t component = parameter;
    if (component.attr<std::string>(_U(name)) == parameter_name) {
      value = component.attr<std::string>(_U(value));
      return true;
    }
  }
  return false;
}

double get_double_parameter(xml_h xml_handle, const char* parameter_name, double default_value) {
  std::string value;
  if (!find_parameter_value(xml_handle, parameter_name, value)) {
    return default_value;
  }
  return dd4hep::_toDouble(value.c_str());
}

int get_int_parameter(xml_h xml_handle, const char* parameter_name, int default_value) {
  std::string value;
  if (!find_parameter_value(xml_handle, parameter_name, value)) {
    return default_value;
  }
  return static_cast<int>(dd4hep::_toDouble(value.c_str()));
}

std::string get_string_parameter(
    xml_h xml_handle,
    const char* parameter_name,
    const std::string& default_value) {
  std::string value;
  if (!find_parameter_value(xml_handle, parameter_name, value)) {
    return default_value;
  }
  return value;
}

GeoParameters read_parameters(xml_h xml_handle) {
  GeoParameters geo_parameters;
  geo_parameters.half_width_x =
      get_double_parameter(xml_handle, "dx", geo_parameters.half_width_x);
  geo_parameters.half_width_y =
      get_double_parameter(xml_handle, "dy", geo_parameters.half_width_y);
  geo_parameters.front_face_z =
      get_double_parameter(xml_handle, "zmin", geo_parameters.front_face_z);
  geo_parameters.spacer_thickness =
      get_double_parameter(xml_handle, "t_spacer", geo_parameters.spacer_thickness);
  geo_parameters.layer_count =
      get_int_parameter(xml_handle, "nLayers", geo_parameters.layer_count);
  geo_parameters.side = get_string_parameter(xml_handle, "side", geo_parameters.side);
  geo_parameters.absorber_material_name =
      get_string_parameter(
          xml_handle,
          "absorberMaterial",
          geo_parameters.absorber_material_name);
  geo_parameters.active_material_name =
      get_string_parameter(xml_handle, "activeMaterial", geo_parameters.active_material_name);
  geo_parameters.spacer_material_name =
      get_string_parameter(xml_handle, "spacerMaterial", geo_parameters.spacer_material_name);

  const std::array<const char*, kSegmentCount> segment_layer_keys {{
      "seg1_layers", "seg2_layers", "seg3_layers"
  }};
  const std::array<const char*, kSegmentCount> segment_absorber_keys {{
      "t_absorber_seg1", "t_absorber_seg2", "t_absorber_seg3"
  }};
  const std::array<const char*, kSegmentCount> segment_scintillator_keys {{
      "t_scin_seg1", "t_scin_seg2", "t_scin_seg3"
  }};

  for (int segment_index = 0; segment_index < kSegmentCount; ++segment_index) {
    geo_parameters.segment_layer_counts[segment_index] =
        get_int_parameter(
            xml_handle,
            segment_layer_keys[segment_index],
            geo_parameters.segment_layer_counts[segment_index]);
    geo_parameters.segment_absorber_thicknesses[segment_index] =
        get_double_parameter(
            xml_handle,
            segment_absorber_keys[segment_index],
            geo_parameters.segment_absorber_thicknesses[segment_index]);
    geo_parameters.segment_scintillator_thicknesses[segment_index] =
        get_double_parameter(
            xml_handle,
            segment_scintillator_keys[segment_index],
            geo_parameters.segment_scintillator_thicknesses[segment_index]);
  }

  return geo_parameters;
}

std::vector<ResolvedSegment> resolve_segments(
    const GeoParameters& geo_parameters,
    double minimum_build_thickness) {
  int segment_layer_sum = 0;

  for (int segment_index = 0; segment_index < kSegmentCount; ++segment_index) {
    segment_layer_sum += geo_parameters.segment_layer_counts[segment_index];
  }

  if (geo_parameters.layer_count <= 0) {
    dd4hep::printout(dd4hep::FATAL, kPluginName, "nLayers must be positive.");
    throw std::runtime_error("Invalid nLayers");
  }

  if (geo_parameters.half_width_x <= 0.0 || geo_parameters.half_width_y <= 0.0) {
    dd4hep::printout(dd4hep::FATAL, kPluginName, "dx and dy must be positive.");
    throw std::runtime_error("Invalid detector width");
  }

  for (int segment_index = 0; segment_index < kSegmentCount; ++segment_index) {
    if (geo_parameters.segment_layer_counts[segment_index] <= 0) {
      dd4hep::printout(
          dd4hep::FATAL,
          kPluginName,
          "seg%d_layers must be positive.",
          segment_index + 1);
      throw std::runtime_error("Invalid segment layer count");
    }
  }

  if (segment_layer_sum != geo_parameters.layer_count) {
    dd4hep::printout(
        dd4hep::FATAL,
        kPluginName,
        "seg1_layers + seg2_layers + seg3_layers must equal nLayers.");
    throw std::runtime_error("Segment layer count mismatch");
  }

  std::vector<ResolvedSegment> resolved_segments;
  resolved_segments.reserve(kSegmentCount);
  for (int segment_index = 0; segment_index < kSegmentCount; ++segment_index) {
    const double absorber_thickness =
        geo_parameters.segment_absorber_thicknesses[segment_index];
    const double spacer_thickness = geo_parameters.spacer_thickness;
    const double scintillator_thickness =
        geo_parameters.segment_scintillator_thicknesses[segment_index];

    resolved_segments.push_back(ResolvedSegment {
        geo_parameters.segment_layer_counts[segment_index],
        absorber_thickness,
        spacer_thickness,
        scintillator_thickness
    });
  }
  return resolved_segments;
}

double get_detector_thickness(
    const std::vector<ResolvedSegment>& resolved_segments) {
  double detector_thickness = 0.0;

  for (const ResolvedSegment& resolved_segment : resolved_segments) {
    if (resolved_segment.layer_count <= 0) {
      dd4hep::printout(dd4hep::FATAL, kPluginName, "Segment layer count must be positive.");
      throw std::runtime_error("Invalid segment layer count");
    }

    const double layer_thickness =
        resolved_segment.absorber_thickness +
        resolved_segment.scintillator_thickness +
        2.0 * resolved_segment.spacer_thickness;
    detector_thickness += resolved_segment.layer_count * layer_thickness;
  }

  if (detector_thickness <= 0.0) {
    dd4hep::printout(dd4hep::FATAL, kPluginName, "Detector thickness must be positive.");
    throw std::runtime_error("Invalid detector thickness");
  }

  return detector_thickness;
}

Material require_material(Detector& detector, const std::string& material_name) {
  try {
    return detector.material(material_name);
  } catch (...) {
    dd4hep::printout(
        dd4hep::FATAL,
        kPluginName,
        "Material '%s' not found.",
        material_name.c_str());
    throw;
  }
}

std::vector<SegmentVolumes> build_segment_volumes(
    SensitiveDetector& sensitive_detector,
    const std::string& detector_name,
    const GeoParameters& geo_parameters,
    const std::vector<ResolvedSegment>& resolved_segments,
    Material absorber_material,
    Material scintillator_material,
    Material spacer_material,
    const VisAttr& absorber_vis,
    const VisAttr& scintillator_vis,
    const VisAttr& spacer_vis) {
  std::vector<SegmentVolumes> segment_volumes;
  segment_volumes.reserve(resolved_segments.size());

  for (std::size_t segment_index = 0; segment_index < resolved_segments.size(); ++segment_index) {
    const ResolvedSegment& resolved_segment = resolved_segments[segment_index];
    SegmentVolumes segment_volume;
    segment_volume.layer_count = resolved_segment.layer_count;
    segment_volume.absorber_half_thickness = 0.5 * resolved_segment.absorber_thickness;
    segment_volume.scintillator_half_thickness = 0.5 * resolved_segment.scintillator_thickness;
    segment_volume.spacer_half_thickness = 0.5 * resolved_segment.spacer_thickness;

    const std::string segment_suffix = "_seg" + std::to_string(segment_index + 1);
    segment_volume.absorber_volume = Volume(
        detector_name + "_abs" + segment_suffix,
        Box(
            geo_parameters.half_width_x,
            geo_parameters.half_width_y,
            segment_volume.absorber_half_thickness),
        absorber_material);
    segment_volume.scintillator_volume = Volume(
        detector_name + "_scin" + segment_suffix,
        Box(
            geo_parameters.half_width_x,
            geo_parameters.half_width_y,
            segment_volume.scintillator_half_thickness),
        scintillator_material);

    if (absorber_vis.isValid()) {
      segment_volume.absorber_volume.setVisAttributes(absorber_vis);
    }
    if (scintillator_vis.isValid()) {
      segment_volume.scintillator_volume.setVisAttributes(scintillator_vis);
    }
    segment_volume.scintillator_volume.setSensitiveDetector(sensitive_detector);

    segment_volume.spacer_volume = Volume(
        detector_name + "_spacer" + segment_suffix,
        Box(
            geo_parameters.half_width_x,
            geo_parameters.half_width_y,
            segment_volume.spacer_half_thickness),
        spacer_material);
    if (spacer_vis.isValid()) {
      segment_volume.spacer_volume.setVisAttributes(spacer_vis);
    }

    segment_volumes.push_back(segment_volume);
  }

  return segment_volumes;
}

}  // namespace

static Ref_t factory(Detector& detector, xml_h xml_handle, SensitiveDetector sensitive_detector) {
  xml_det_t detector_xml = xml_handle;
  const std::string detector_name = detector_xml.nameStr();
  const double minimum_build_thickness = 0.01 * dd4hep::mm;

  GeoParameters geo_parameters = read_parameters(xml_handle);
  const std::vector<ResolvedSegment> resolved_segments =
      resolve_segments(geo_parameters, minimum_build_thickness);
  const double detector_thickness = get_detector_thickness(resolved_segments);
  const double half_detector_thickness = 0.5 * detector_thickness;
  const bool place_positive_side = geo_parameters.side != "-z";
  const double detector_center_z =
      place_positive_side
          ? geo_parameters.front_face_z + half_detector_thickness
          : -(geo_parameters.front_face_z + half_detector_thickness);

  const Material air_material = require_material(detector, "Air");
  const Material absorber_material =
      require_material(detector, geo_parameters.absorber_material_name);
  const Material scintillator_material =
      require_material(detector, geo_parameters.active_material_name);
  const Material spacer_material =
      require_material(detector, geo_parameters.spacer_material_name);

  const VisAttr detector_vis = detector.visAttributes("HCALVis");
  const VisAttr spacer_vis = detector.visAttributes("SpacerVis");
  const VisAttr scintillator_vis = detector.visAttributes("HCalActiveVis");
  const VisAttr absorber_vis = detector.visAttributes("AbsorberVis");

  sensitive_detector.setType("calorimeter");

  DetElement detector_element(detector_name, detector_xml.id());
  Volume mother_volume = detector.pickMotherVolume(detector_element);
  Volume detector_volume(
      detector_name + "_vol",
      Box(
          geo_parameters.half_width_x,
          geo_parameters.half_width_y,
          half_detector_thickness),
      air_material);
  if (detector_vis.isValid()) {
    detector_volume.setVisAttributes(detector_vis);
  }

  const std::vector<SegmentVolumes> segment_volumes = build_segment_volumes(
      sensitive_detector,
      detector_name,
      geo_parameters,
      resolved_segments,
      absorber_material,
      scintillator_material,
      spacer_material,
      absorber_vis,
      scintillator_vis,
      spacer_vis);

  // Advance by half-thickness so each slice is placed at its center.
  double local_z = -half_detector_thickness;
  int layer_index = 0;
  for (const SegmentVolumes& segment_volume : segment_volumes) {
    for (int segment_layer_index = 0;
         segment_layer_index < segment_volume.layer_count;
         ++segment_layer_index, ++layer_index) {
      local_z += segment_volume.absorber_half_thickness;
      detector_volume.placeVolume(segment_volume.absorber_volume, Position(0, 0, local_z))
          .addPhysVolID("layer", layer_index)
          .addPhysVolID("slice", 0);
      local_z += segment_volume.absorber_half_thickness;

      local_z += segment_volume.spacer_half_thickness;
      detector_volume.placeVolume(segment_volume.spacer_volume, Position(0, 0, local_z))
          .addPhysVolID("layer", layer_index)
          .addPhysVolID("slice", 1);
      local_z += segment_volume.spacer_half_thickness;

      local_z += segment_volume.scintillator_half_thickness;
      detector_volume.placeVolume(segment_volume.scintillator_volume, Position(0, 0, local_z))
          .addPhysVolID("layer", layer_index)
          .addPhysVolID("slice", 2);
      local_z += segment_volume.scintillator_half_thickness;

      local_z += segment_volume.spacer_half_thickness;
      detector_volume.placeVolume(segment_volume.spacer_volume, Position(0, 0, local_z))
          .addPhysVolID("layer", layer_index)
          .addPhysVolID("slice", 3);
      local_z += segment_volume.spacer_half_thickness;
    }
  }

  PlacedVolume detector_placement =
      mother_volume.placeVolume(detector_volume, Position(0, 0, detector_center_z));
  detector_placement.addPhysVolID("system", detector_xml.id());
  detector_element.setPlacement(detector_placement);

  return detector_element;
}

DECLARE_DETELEMENT(hcal, factory)
