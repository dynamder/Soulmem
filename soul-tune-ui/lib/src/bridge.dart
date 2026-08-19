/// FRB 桥接封装：初始化 Rust 运行时，并把生成的底层 API 转成强类型模型流。
///
/// 注意：`rust/api.dart` 与 `rust/frb_generated.dart` 由
/// `flutter_rust_bridge_codegen generate` 生成（见 flutter_rust_bridge.yaml）。
/// 本文件假设生成面为：
///   `scanDatasetsJson(dir)` -> Future
///   `datasetMetaJson(path)` -> Future
///   `defaultParamsJson()` -> Future
///   `resetCancel()` -> Future
///   `runSuite({algo, dataset, paramsJson})` -> Stream
///   `runBatch({dir, mode, paramsJson})` -> Stream
library;

import 'dart:convert';

import 'models.dart';
import 'rust/api.dart' as api;
import 'rust/frb_generated.dart' as rust;

Future<void> initRust() async {
  await rust.RustLib.init();
}

Future<List<DatasetEntry>> scanDatasets(String dir) async {
  final raw = await api.scanDatasetsJson(dir: dir);
  if (raw.trim().isEmpty) return [];
  final list = (jsonDecode(raw) as List).cast<Map<String, dynamic>>();
  return list.map(DatasetEntry.fromJson).toList();
}

Future<DatasetMeta> datasetMeta(String path) async {
  final raw = await api.datasetMetaJson(path: path);
  return DatasetMeta.fromJson(jsonDecode(raw) as Map<String, dynamic>);
}

Future<List<ParamSpec>> defaultParams() async {
  final raw = await api.defaultParamsJson();
  final list = (jsonDecode(raw) as List).cast<Map<String, dynamic>>();
  return list.map(ParamSpec.fromJson).toList();
}

Future<void> resetCancel() => api.resetCancel();

Stream<RunEvent> runSuite({
  required String algo,
  required String dataset,
  required Map<String, String> params,
}) {
  return api
      .runSuite(
        algo: algo,
        dataset: dataset,
        paramsJson: _encodeParams(params),
      )
      .map((raw) => RunEvent.fromJson(_decode(raw)));
}

Stream<BatchEvent> runBatch({
  required String dir,
  required String mode,
  required Map<String, String> params,
}) {
  return api
      .runBatch(
        dir: dir,
        mode: mode,
        paramsJson: _encodeParams(params),
      )
      .map((raw) => BatchEvent.fromJson(_decode(raw)));
}

Stream<CompareEvent> runCompare({
  required String dataset,
  required Map<String, String> params,
}) {
  return api
      .runCompare(
        dataset: dataset,
        paramsJson: _encodeParams(params),
      )
      .map((raw) => CompareEvent.fromJson(_decode(raw)));
}

Future<InspectFile> inspectFile(String path) async {
  final raw = await api.inspectFileJson(path: path);
  return InspectFile.fromJson(jsonDecode(raw) as Map<String, dynamic>);
}

Future<InspectEntries> inspectEntries(String path) async {
  final raw = await api.inspectEntriesJson(path: path);
  return InspectEntries.fromJson(jsonDecode(raw) as Map<String, dynamic>);
}

Stream<ForgetEvent> runForget({
  required String mode,
  required String dataset,
}) {
  return api
      .runForget(mode: mode, dataset: dataset)
      .map((raw) => ForgetEvent.fromJson(_decode(raw)));
}

Future<PlaytestStartResult> playtestStart(String graphDir) async {
  final raw = await api.playtestStart(graphDir: graphDir);
  return PlaytestStartResult.fromJson(jsonDecode(raw) as Map<String, dynamic>);
}

Future<void> playtestFinish() => api.playtestFinish();

Stream<PlayTurn> playtestTurn(String userMessage) {
  return api
      .playtestTurn(userMessage: userMessage)
      .map((raw) => PlayTurn.fromJson(_decode(raw)));
}

String _encodeParams(Map<String, String> params) => jsonEncode(params);

Map<String, dynamic> _decode(String raw) => jsonDecode(raw) as Map<String, dynamic>;
