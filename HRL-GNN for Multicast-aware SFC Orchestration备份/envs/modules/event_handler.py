class EventHandler:
    """事件处理器：处理请求离开与资源回收"""

    def __init__(self, resource_manager):
        self.resource_mgr = resource_manager
        # 记录正在服务的请求状态: req_id -> {paths: [], vnfs: [(node, cpu, mem)]}
        self.active_services = {}

    def register_service(self, req_id: int, deploy_info: dict):
        """注册已部署的服务 (用于后续回收)"""
        # deploy_info 结构需包含占用的路径和节点资源
        self.active_services[req_id] = deploy_info

    def process_leaves(self, leave_ids: list):
        """处理离开事件，回收资源"""
        for rid in leave_ids:
            if rid in self.active_services:
                info = self.active_services.pop(rid)
                self._release_resources(info)

    def _release_resources(self, info: dict):
        """释放具体资源"""
        # 释放路径带宽
        for path in info.get('paths', []):
            bw = info.get('bw', 0.0)
            self.resource_mgr.release_link_resource(path, bw)

        # 释放VNF计算资源
        for vnf in info.get('vnfs', []):
            node_id, cpu, mem = vnf
            self.resource_mgr.release_node_resource(node_id, cpu, mem)

    def reset(self):
        self.active_services.clear()